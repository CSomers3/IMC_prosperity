## Import Python libraries
from copy import deepcopy
import pandas as pd

## Import the file you will upload
import Round_1.submissionEMA as Algo

## Import our custom PnLEstimator
import local_PnLEstimator as PnLEstimator
from local_playground_suppress_print_context_manager import suppress_output

## Import the IMC classes
from datamodel import Order, TradingState, Symbol, OrderDepth


SUPPRESS_PRINTS: bool = False
ROUND = 4


def run_pnl_estimation(
        bananas_best_average_profit,
        pearls_best_average_profit,
        coconuts_best_average_profit,
        pina_coladas_best_average_profit,
        berries_best_average_profit,
        diving_gear_best_average_profit,
        dip_best_average_profit,
        baguette_best_average_profit,
        ukulele_best_average_profit,
        picnic_basket_best_average_profit,
        min_profit,
        min_spread,
        ema_short_period,
        ema_long_period,
        data,
        data_trades,
):
    # for product in "BANANAS", "PEARLS", "BERRIES", "COCONUTS", "PINA_COLADAS", "DIVING_GEAR":
    #     if product in Algo.MIN_PROFIT:
    #         Algo.MIN_PROFIT[product] = min_profit
    #     if product in Algo.SPREAD_TO_MM:
    #         Algo.SPREAD_TO_MM[product] = min_spread
    #     if product in Algo.EMA_SHORT_PERIOD:
    #         # Them two go together
    #         Algo.EMA_SHORT_PERIOD[product] = ema_short_period
    #         Algo.EMA_LONG_PERIOD[product] = ema_long_period

    ## Loop through the historic days
    all_profits: list[dict[str, float]] = []
    day: str
    for day in [
        "1",
        "2",
        "3",
    ]:
        print("=====================================")
        print(f"RUNNING DAY {day}")
        print("=====================================")

        df_simulation: pd.DataFrame = data[f"prices_round_{ROUND}_day_{day}.csv"]
        trades = data_trades[f"trades_round_{ROUND}_day_{day}_nn.csv"]

        # ## Only take the first % of the rows of df_simulation
        # df_simulation = df_simulation[:int(len(df_simulation) * 0.1)]

        ## Create the Trader object
        trader = Algo.Trader()

        # Print hyperparameters
        with suppress_output(SUPPRESS_PRINTS):
            print("FAIR_VALUE_SHIFT_AT_CROSSOVER:", Algo.FAIR_VALUE_SHIFT_AT_CROSSOVER)
            print("MIN_PROFIT:", Algo.MIN_PROFIT)
            print("SPREAD_TO_MM:", Algo.SPREAD_TO_MM)
            print("EMA_SHORT_PERIOD:", Algo.EMA_SHORT_PERIOD)
            print("EMA_LONG_PERIOD:", Algo.EMA_LONG_PERIOD)

        ## List of all products is the list without doublons of df["product"]
        products = list(df_simulation["product"].unique())
        with suppress_output(SUPPRESS_PRINTS):
            print("PRODUCTS TESTED IN THE PLAYGROUND:", products)

        ## Drop dolphins
        products.remove("DOLPHIN_SIGHTINGS")

        pnl_estimator = PnLEstimator.ProfitsAndLossesEstimator(products)

        ## Every row of df_simulation, we need to call the trader's method Trader.run()
        ## The trader's method Trader.run() takes one argument, an object TradingState, that we need to
        ## build
        old_trading_state: TradingState = TradingState(
            timestamp=-1,
            listings={},
            order_depths={},
            own_trades={},
            market_trades={},
            position={product: 0 for product in products},
            observations={},
        )
        for timestamp in df_simulation["timestamp"].unique():
            # To build a TradingState object, we need to build a dictionnary of OrderDepth objects
            # For each product, we need to build an OrderDepth object
            order_depths: dict[Symbol, OrderDepth] = {}
            for product in products:
                order_depth: OrderDepth = OrderDepth()
                # Find the first row of df_simulation where the product is the one we are looking for
                # (right product and right timestamp)
                row: pd.DataFrame = df_simulation[
                    (df_simulation["product"] == product)
                    & (df_simulation["timestamp"] == timestamp)
                    ]
                ## Comment out for speed
                assert (
                        len(row) == 1
                )  # We should have only one row in our dataframe
                ##
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
                market_trades={
                    product: []
                    for product in products
                },
                position=old_trading_state.position,
                observations={
                    'DOLPHIN_SIGHTINGS':
                    df_simulation[
                        (df_simulation['timestamp'] == timestamp)
                        &
                        (df_simulation['product'] == 'DOLPHIN_SIGHTINGS')
                    ]["mid_price"].values[0]
                },
            )

            orders: dict[str, list[Order]]
            with suppress_output(SUPPRESS_PRINTS):
                orders = trader.run(
                    deepcopy(current_trading_state)
                )  # get the orders from the trader

            # Update the positions & PnL for the trades that go through (no market making here)
            for product in products:
                if product in orders:
                    best_bids: list[tuple[int, int]]
                    best_asks: list[tuple[int, int]]
                    best_bids, best_asks, spread = Algo.get_top_of_book(
                        current_trading_state.order_depths[product]
                    )
                    order: Order
                    for order in orders[product]:
                        if order.symbol != product:
                            raise ValueError(
                                "Order symbol does not match product"
                            )
                        else:
                            if order.quantity > 0:
                                # It's a Buy order, so we look at the asks
                                for idx, (bid, vol) in enumerate(best_asks):
                                    if order.price >= bid:
                                        if (
                                                order.quantity > -vol
                                        ):  # -vol is positive
                                            # we buy what we can, and the rest is left for the bots to
                                            # potentially trade on
                                            current_trading_state.position[
                                                product
                                            ] += -vol
                                            pnl_estimator.update(
                                                order, partial=-vol
                                            )
                                            order.quantity -= -vol
                                            best_asks[idx] = (
                                                bid,
                                                best_asks[idx][1] - vol,
                                            )  # update the volume
                                        else:
                                            # we buy all the quantity we have put in the Buy order
                                            current_trading_state.position[
                                                product
                                            ] += order.quantity
                                            pnl_estimator.update(order)
                                            best_asks[idx] = (
                                                bid,
                                                best_asks[idx][1]
                                                + order.quantity,
                                            )  # update the volume
                                            order.quantity = 0
                            else:
                                # It's a Sell Order, so we look at the bids
                                for idx, (ask, vol) in enumerate(best_bids):
                                    if order.price <= ask:
                                        if (
                                                order.quantity < -vol
                                        ):  # e.g. -5 < -2 (vol is positive)
                                            # we sell what we can, and the rest is left for the bots to
                                            # potentially trade on
                                            current_trading_state.position[
                                                product
                                            ] -= vol
                                            pnl_estimator.update(
                                                order, partial=-vol
                                            )
                                            order.quantity += vol
                                            best_bids[idx] = (
                                                ask,
                                                best_bids[idx][1] - vol,
                                            )  # update the volume
                                        else:
                                            # we sell all the quantity we have put in the Sell order
                                            current_trading_state.position[
                                                product
                                            ] += order.quantity
                                            pnl_estimator.update(order)
                                            best_bids[idx] = (
                                                ask,
                                                best_bids[idx][1]
                                                + order.quantity,
                                            )  # update the volume
                                            order.quantity = 0

            # Update the positions & PnL for the trades that don't go through (market making here)
            for product in products:
                if product in orders:
                    order: Order
                    for order in orders[product]:
                        if order.quantity != 0:
                            # The order did not go through, so we market make
                            if order.quantity > 0:
                                # let's check in data_trades[f"trades_round_{ROUND}_day_{day}"] if we find a
                                # trade with the right timestamp and the right product and the right
                                # price (it's a buy order here, so we want to find trades that were made
                                # at a lower price than the one we are). If we find one, we add the
                                # quantity to our position and update the PnL
                                trades_filtered = trades[trades["symbol"] == product]
                                trades_filtered = trades_filtered[
                                    trades_filtered["timestamp"] == timestamp
                                ]
                                trades_filtered = trades_filtered[
                                    trades_filtered["price"] <= order.price
                                ]
                                # if it's equal, we get it because we never MM at the same price as a possible trade
                                if len(trades_filtered) > 0:
                                    # we buy what we can
                                    volume_traded = min(
                                        order.quantity,
                                        trades_filtered["quantity"].sum(),
                                    )
                                    current_trading_state.position[
                                        product
                                    ] += volume_traded
                                    pnl_estimator.update(
                                        order, partial=volume_traded
                                    )
                                    order.quantity -= -volume_traded
                            elif order.quantity < 0:
                                # let's check in data_trades[f"trades_round_{ROUND}_day_{day}"] if we find a
                                # trade with the right timestamp and the right product and the right
                                # price (it's a sell order here, so we want to find trades that were
                                # made at a higher price than the one we are) If we find one, we add the
                                # quantity to our position and update the PnL
                                trades_filtered = trades[trades["symbol"] == product]
                                trades_filtered = trades_filtered[
                                    trades_filtered["timestamp"] == timestamp
                                ]
                                trades_filtered = trades_filtered[
                                    trades_filtered["price"] >= order.price
                                ]  # if it's equal, we get it because
                                # we never MM at the same price as a possible trade
                                if len(trades_filtered) > 0:
                                    # we sell what we can
                                    volume_traded = min(
                                        -order.quantity,
                                        trades_filtered["quantity"].sum(),
                                    )
                                    current_trading_state.position[
                                        product
                                    ] -= volume_traded
                                    pnl_estimator.update(
                                        order, partial=-volume_traded
                                    )
                                    order.quantity -= volume_traded

            ## COMMENT OUT FOR SPEED
            # Sanity check
            pos_limit = {
                "PEARLS": 20,
                "BANANAS": 20,
                "COCONUTS": 600,
                "PINA_COLADAS": 300,
                "BERRIES": 250,
                "DIVING_GEAR": 50,
                "BAGUETTE": 150,
                "DIP": 300,
                "UKULELE": 70,
                "PICNIC_BASKET": 70
            }
            for product in products:
                try:
                    assert current_trading_state.position[product] <= pos_limit[product]
                    assert current_trading_state.position[product] >= -pos_limit[product]
                except AssertionError:
                    breakpoint()
            ##

            # Liquidate positions at the end of the timestamp to have an accurate estimation of PnL at
            # the end of the timestamp
            if current_trading_state.timestamp == 999900:
                all_profits_and_losses: dict[Symbol, int] = deepcopy(
                    pnl_estimator.get_all()
                )
                for product in products:
                    # get last mid price
                    order_depth = order_depths[product]
                    best_bid: int = max(order_depth.buy_orders.keys())
                    best_ask: int = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    # simulate liquidation
                    all_profits_and_losses[product] += (
                            current_trading_state.position[product] * mid_price
                    )
                    with suppress_output(SUPPRESS_PRINTS):
                        print(
                            f"{timestamp}: SEASHELLS AFTER LIQUIDATION PRODUCT "
                            f"{product} {all_profits_and_losses[product]}"
                        )

            old_trading_state = current_trading_state

        # Print the results
        with suppress_output(SUPPRESS_PRINTS):
            print(
                "PROFITS:", all_profits_and_losses  # noqa, there is at least one timestamp in the simulation
            )
            print("")
        all_profits.append(all_profits_and_losses)

    with suppress_output(SUPPRESS_PRINTS):
        print("\n========================================")
        print("========================================")
        print("========================================\n")

    for product in products:
        with suppress_output(SUPPRESS_PRINTS):
            print(f"PRODUCT {product}")
            print(
                "PROFITS:",
                [all_profits[i][product] for i in range(len(all_profits))],
            )
            print(
                "AVERAGE PROFIT:",
                sum([all_profits[i][product] for i in range(len(all_profits))])
                / len(all_profits),
            )
            print("")

    best_average_profits = {
        "BANANAS": bananas_best_average_profit,
        "PEARLS": pearls_best_average_profit,
        "COCONUTS": coconuts_best_average_profit,
        "PINA_COLADAS": pina_coladas_best_average_profit,
        "BERRIES": berries_best_average_profit,
        "DIVING_GEAR": diving_gear_best_average_profit,
        "DIP": dip_best_average_profit,
        "BAGUETTE": baguette_best_average_profit,
        "UKULELE": ukulele_best_average_profit,
        "PICNIC_BASKET": picnic_basket_best_average_profit,
    }
    for item, best_average_profit in best_average_profits.items():
        new_average_score: float = sum(all_profits[i][item] for i in range(len(all_profits))) / len(all_profits)
        if best_average_profit[1] < new_average_score:
            best_average_profit[0] = (
                f"MIN_PROFIT = {min_profit}, "
                f"SPREAD_TO_MM = {min_spread}, "
                f"EMA_SHORT_PERIOD = {ema_short_period}, "
                f"EMA_LONG_PERIOD = {ema_long_period}"
            )
            best_average_profit[1] = round(new_average_score, 0)
            best_average_profit[2] = [round(all_profits[i][item], 0) for i in range(len(all_profits))]
            print(f"NEW {item} HIGHSCORE", best_average_profit)
