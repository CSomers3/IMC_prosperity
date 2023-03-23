## Import Python Libraries
import os
import pandas as pd
from copy import deepcopy
## Import the file you will upload
import Round_1.submissionEMA as Algo
## Import our custom PnLEstimator
import local_PnLEstimator as PnLEstimator
from local_playground_suppress_print_context_manager import suppress_output
## Import the IMC classes
from datamodel import Order, TradingState, Symbol, OrderDepth


SUPPRESS_PRINTS: bool = False


if __name__ == "__main__":

    ## Open all csvs in the folder data and save them into a dictionary (filename = key, df = value)
    data: dict[str, pd.DataFrame] = {}
    for file in os.listdir("Round_1/data"):
        if file.endswith(".csv") and file.startswith("prices"):
            data[file] = pd.read_csv("Round_1/data/" + file, sep=";")

    ## Loop through the historic days
    day: str
    for day in "-1", "0":
        print("=====================================")
        print(f"DATA FOR DAY {day}")
        print("=====================================")

        df_simulation: pd.DataFrame = data[f"prices_round_1_day_{day}.csv"]


        ## Create the Trader object
        trader = Algo.Trader()

        # Print hyperparameters
        print("FAIR_VALUE_SHIFT_AT_CROSSOVER:", Algo.FAIR_VALUE_SHIFT_AT_CROSSOVER)
        print("TIME_WHILST_USING_DEFAULT_FAIR_VALUE:", Algo.TIME_WHILST_USING_DEFAULT_FAIR_VALUE)
        print("SPREAD_ADJUSTMENT:", Algo.SPREAD_ADJUSTMENT)

        ## List of all products is the list without doublons of df["product"]
        products = list(df_simulation["product"].unique())
        print("PRODUCTS TESTED IN THE PLAYGROUND:", products)

        pnl_estimator = PnLEstimator.ProfitsAndLossesEstimator(products)

        ## Every row of df_simulation, we need to call the trader's method Trader.run()
        ## The trader's method Trader.run() takes one argument, an object TradingState, that we need to build
        old_trading_state: TradingState = TradingState(
            timestamp=-1,
            listings={},
            order_depths={},
            own_trades={},
            market_trades={},
            position={product: 0 for product in products},
            observations={}
        )
        for timestamp in df_simulation["timestamp"].unique():
            # To build a TradingState object, we need to build a dictionnary of OrderDepth objects
            # For each product, we need to build an OrderDepth object
            order_depths: dict[Symbol, OrderDepth] = {}
            for product in products:
                order_depth: OrderDepth = OrderDepth()
                # Find the first row of df_simulation where the product is the one we are looking for (right product
                # and right timestamp)
                row: pd.DataFrame = df_simulation[
                    (df_simulation["product"] == product) & (df_simulation["timestamp"] == timestamp)
                ]
                assert len(row) == 1  # We should have only one row in our dataframe
                order_depth.buy_orders = {
                    int(row[f"bid_price_{i}"]): int(row[f"bid_volume_{i}"])
                    for i in range(1, 4)
                    if not pd.isna(row[f"bid_price_{i}"]).any()
                }
                order_depth.sell_orders = {
                    int(row[f"ask_price_{i}"]): -int(row[f"ask_volume_{i}"])
                    for i in range(1, 4)
                    if not pd.isna(row[f"ask_price_{i}"]).any()
                }
                order_depths[product] = order_depth
            current_trading_state = TradingState(
                timestamp=timestamp,
                listings={},  # is not used anyway
                order_depths=order_depths,
                own_trades={},  # is not used anyway
                market_trades={},  # is not used anyway
                position=old_trading_state.position,
                observations={}  # is not used anyway
            )
            orders: dict[str, list[Order]]
            with suppress_output(SUPPRESS_PRINTS):
                orders = trader.run(deepcopy(current_trading_state))  # get the orders from the trader

            # Update the positions for the trades that go through (no market making here)
            for product in products:
                if product in orders:
                    best_bids: list[tuple[int, int]]
                    best_asks: list[tuple[int, int]]
                    best_bids, best_asks, spread = Algo.get_top_of_book(current_trading_state.order_depths[product])
                    order: Order
                    for order in orders[product]:
                        if order.symbol != product:
                            raise ValueError("Order symbol does not match product")
                        else:
                            if order.quantity > 0:
                                # It's a Buy order, so we look at the asks
                                for idx, (bid, vol) in enumerate(best_asks):
                                    if order.price >= bid:
                                        if order.quantity > -vol:  # -vol is positive
                                            # we buy what we can, and the rest is left for the bots to potentially
                                            # trade on
                                            current_trading_state.position[product] += -vol
                                            pnl_estimator.update(order, partial=-vol)
                                            order.quantity -= -vol
                                            best_asks[idx] = (
                                                bid,
                                                best_asks[idx][1] - vol
                                            )  # update the volume
                                        else:
                                            # we buy all the quantity we have put in the Buy order
                                            current_trading_state.position[product] += order.quantity
                                            pnl_estimator.update(order)
                                            best_asks[idx] = (
                                                bid,
                                                best_asks[idx][1] + order.quantity
                                            )  # update the volume
                                            order.quantity = 0
                            else:
                                # It's a Sell Order, so we look at the bids
                                for idx, (ask, vol) in enumerate(best_bids):
                                    if order.price <= ask:
                                        if order.quantity < -vol:  # e.g. -5 < -2 (vol is positive)
                                            # we sell what we can, and the rest is left for the bots to potentially
                                            # trade on
                                            current_trading_state.position[product] -= vol
                                            pnl_estimator.update(order, partial=-vol)
                                            order.quantity -= vol
                                            best_bids[idx] = (
                                                ask,
                                                best_bids[idx][1] - vol
                                            )  # update the volume
                                        else:
                                            # we sell all the quantity we have put in the Sell order
                                            current_trading_state.position[product] += order.quantity
                                            pnl_estimator.update(order)
                                            best_bids[idx] = (
                                                ask,
                                                best_bids[idx][1] + order.quantity
                                            )  # update the volume
                                            order.quantity = 0
            # Sanity check
            for product in products:
                try:
                    assert current_trading_state.position[product] <= 20
                    assert current_trading_state.position[product] >= -20
                except:
                    breakpoint()

            # Liquidate positions at the end of the timestamp to have an accurate estimation of PnL at the end
            # of the timestamp
            all_profits_and_losses: dict[
                Symbol, int
            ] = deepcopy(pnl_estimator.get_all())
            if current_trading_state.timestamp <= 1_000_000:
                for product in products:
                    # get last mid price
                    order_depth = order_depths[product]
                    best_bid: int = max(order_depth.buy_orders.keys())
                    best_ask: int = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    # simulate liquidation
                    all_profits_and_losses[product] += current_trading_state.position[product] * mid_price
                    with suppress_output(SUPPRESS_PRINTS):
                        print(
                            f"SEASHELLS AFTER LIQUIDATION PRODUCT {product} {all_profits_and_losses[product]}"
                        )

            old_trading_state = current_trading_state



        ## Print the PnL
        print("PROFITS:", all_profits_and_losses)  # noqa, there is at least one timestamp in the simulation
        print("")
