## Import Python Libraries
import multiprocessing as mp
import os
import pandas as pd
import time

## Import pnl estimation
from local_playground_pnl_estimation import run_pnl_estimation, ROUND


if __name__ == "__main__":
    start_time = time.time()

    ## Open all csvs in the folder data and save them into a dictionary (filename = key, df = value)
    data: dict[str, pd.DataFrame] = {}
    for file in os.listdir(f"Round_{ROUND}/data"):
        if file.endswith(".csv") and file.startswith("prices"):
            data[file] = pd.read_csv(f"Round_{ROUND}/data/" + file, sep=";")

    ## Open trades_round to simulate the MM trades
    data_trades: dict[str, pd.DataFrame] = {}
    for file in os.listdir(f"Round_{ROUND}/data"):
        if file.endswith(".csv") and file.startswith("trades_round"):
            data_trades[file] = pd.read_csv(f"Round_{ROUND}/data/" + file, sep=";")

    ## Best parameters and associated PnL, shared across child processes
    manager = mp.Manager()
    bananas_best_average_profit: list[str | float] = manager.list(["Test", 0])
    pearls_best_average_profit: list[str | float] = manager.list(["Test", 0])
    processes = []

    list_min_profit: list[int] = [
        0,
        # 1,
        # 2,
        # 5
    ]
    list_min_spread: list[int] = [
        3,
        # 4,
        # 5,
        # 6
    ]
    list_of_potential_percent_put_when_mm: list[int] = [
        0,
        # 5,
        # 10,
        # 15,
        # 18,
        # 20
    ]
    list_of_potential_ema_short_period: list[int] = [
        5,
        # 8,
        # 10,
        # 12,
        # 15,
        # 30,
    ]
    list_of_potential_ema_long_period: list[int] = [
        12,
        # 15,
        # 20,
        # 30,
        # 50,
        # 100,
        # 1000,
    ]
    for min_profit in list_min_profit:
        for min_spread in list_min_spread:
            for percent_put_when_mm in list_of_potential_percent_put_when_mm:
                for ema_short_period in list_of_potential_ema_short_period:
                    for ema_long_period in list_of_potential_ema_long_period:
                        if ema_short_period < ema_long_period:
                            process = mp.Process(
                                target=run_pnl_estimation,
                                args=(
                                    bananas_best_average_profit,
                                    pearls_best_average_profit,
                                    min_profit,
                                    min_spread,
                                    ema_short_period,
                                    ema_long_period,
                                    percent_put_when_mm,
                                    data,
                                    data_trades,
                                )
                            )
                            processes.append(process)
                            process.start()
                            # run_pnl_estimation(
                            #     bananas_best_average_profit,
                            #     pearls_best_average_profit,
                            #     min_spread,
                            #     ema_short_period,
                            #     ema_long_period,
                            #     percent_put_when_mm,
                            #     data,
                            #     data_trades,
                            # )

    # Wait for all processes to finish
    for process in processes:
        process.join()


    print("BANANAS BEST AVERAGE PROFIT:", bananas_best_average_profit[1])
    print("PARAMS:", bananas_best_average_profit[0])
    print("========================================")
    print("PEARLS BEST AVERAGE PROFIT:", pearls_best_average_profit[1])
    print("PARAMS:", pearls_best_average_profit[0])

    print("Total time: ", time.time() - start_time, "s")
