import os
import pandas as pd

## Import the file you will upload
import submissionEMA

## Import the IMC classes
from datamodel import Order, TradingState, Symbol, OrderDepth, Listing


if __name__ == "__main__":

    ## Open all csvs in the folder data and save them into a dictionary (filename = key, df = value)
    data: dict[str, pd.DataFrame] = {}
    for file in os.listdir("Round_1/data"):
        if file.endswith(".csv") and file.startswith("prices"):
            data[file] = pd.read_csv("Round_1/data/" + file, sep=";")

    ## Decide what day you will use for the simulation
    day: str

    for day in "-2", "-1", "0":
        print(f"DATA FOR DAY {day}")
        df_simulation: pd.DataFrame = data[f"prices_round_1_day_{day}.csv"]
        for spread_bananas in (0, 0.02, 0.1):
            for spread_pearls in (0, 0.02, 0.1):  # I just realized this is stupid, bananas and pearls are independent
                # so i'm testing spread_bananas = 0 with spread_pearls = 0 then spread_bananas = 0 with
                # spread_pearls = 0.02 etc. But we already know what the PnL for bananas will be.
                submissionEMA.SPREAD_ADJUSTMENT = {
                    "BANANAS": spread_bananas,
                    "PEARLS": spread_pearls,
                }

                # Print hyperparameters
                print("FAIR_VALUE_SHIFT_AT_CROSSOVER:", submissionEMA.FAIR_VALUE_SHIFT_AT_CROSSOVER)
                print("TIME_WHILST_USING_DEFAULT_FAIR_VALUE:", submissionEMA.TIME_WHILST_USING_DEFAULT_FAIR_VALUE)
                print("SPREAD_ADJUSTMENT:", submissionEMA.SPREAD_ADJUSTMENT)

                ## Create the Trader object
                trader = submissionEMA.Trader()

                ## List of all products is the list without doublons of df["product"]
                products = list(df_simulation["product"].unique())

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
                        initial_order_depth = OrderDepth()
                        # Find the first row of df_simulation where the product is the one we are looking for (right product and
                        # right timestamp)
                        row: pd.DataFrame = df_simulation[
                            (df_simulation["product"] == product) & (df_simulation["timestamp"] == timestamp)
                        ]
                        assert len(row) == 1  # We should have only one row in our dataframe
                        initial_order_depth.buy_orders = {
                            int(row[f"bid_price_{i}"]): int(row[f"bid_volume_{i}"])
                            for i in range(1, 4)
                            if not pd.isna(row[f"bid_price_{i}"]).any()
                        }
                        initial_order_depth.sell_orders = {
                            int(row[f"ask_price_{i}"]): -int(row[f"ask_volume_{i}"])
                            for i in range(1, 4)
                            if not pd.isna(row[f"ask_price_{i}"]).any()
                        }
                        order_depths[product] = initial_order_depth
                    current_trading_state = TradingState(
                        timestamp=0,
                        listings={},  # is not used anyway
                        order_depths=order_depths,
                        own_trades={},  # is not used anyway
                        market_trades={},  # is not used anyway
                        position=old_trading_state.position,
                        observations={}  # is not used anyway
                    )
                    orders: dict[str, list[Order]]
                    all_profits_and_losses: dict[Symbol, int]
                    orders, all_profits_and_losses = trader.run(current_trading_state)

                    # Update the positions, the PnL are calculated in the submission file
                    for product in products:
                        if product in orders:
                            for order in orders[product]:
                                current_trading_state.position[product] += order.quantity

                    # Sanity check
                    for product in products:
                        assert current_trading_state.position[product] <= 20
                        assert current_trading_state.position[product] >= -20

                    old_trading_state = current_trading_state

                ## Print the PnL
                print(all_profits_and_losses)  # noqa, there is at least one timestamp in the simulation
                print("\n")
