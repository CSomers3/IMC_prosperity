## Import Python Libraries
import multiprocessing as mp
import os
import pandas as pd
import time

## Import pnl estimation
from local_playground_pnl_estimation import run_pnl_estimation, ROUND


# Function to be run in parallel
def run_pnl_estimation_wrapper(args):
    return run_pnl_estimation(*args)


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
    bananas_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    pearls_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    coconuts_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    pina_coladas_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    berries_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    diving_gear_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    dip_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    baguette_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    ukulele_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
    picnic_basket_best_average_profit: list[str | float | list[float]] = manager.list(["Test", -500000, []])
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

    # Create a list of arguments for each process
    process_args = []
    for min_profit in list_min_profit:
        for min_spread in list_min_spread:
            for ema_short_period in list_of_potential_ema_short_period:
                for ema_long_period in list_of_potential_ema_long_period:
                    if ema_short_period < ema_long_period:
                        args = (
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
                        )
                        process_args.append(args)

    # Set the maximum number of concurrent processes
    max_processes = 40

    # Create a process pool and run the processes
    with mp.Pool(processes=max_processes) as pool:
        pool.map(run_pnl_estimation_wrapper, process_args)


    print("BANANAS BEST AVERAGE PROFIT:", bananas_best_average_profit[1])
    print("PARAMS:", bananas_best_average_profit[0])
    print("========================================")
    print("PEARLS BEST AVERAGE PROFIT:", pearls_best_average_profit[1])
    print("PARAMS:", pearls_best_average_profit[0])
    print("========================================")
    print("COCONUTS BEST AVERAGE PROFIT:", coconuts_best_average_profit[1])
    print("PARAMS:", coconuts_best_average_profit[0])
    print("========================================")
    print("PINA COLADAS BEST AVERAGE PROFIT:", pina_coladas_best_average_profit[1])
    print("PARAMS:", pina_coladas_best_average_profit[0])
    print("========================================")
    print("BERRIES BEST AVERAGE PROFIT:", berries_best_average_profit[1])
    print("PARAMS:", berries_best_average_profit[0])
    print("========================================")
    print("DIVING GEAR BEST AVERAGE PROFIT:", diving_gear_best_average_profit[1])
    print("PARAMS:", diving_gear_best_average_profit[0])
    print("========================================")
    print("DIP BEST AVERAGE PROFIT:", dip_best_average_profit[1])
    print("PARAMS:", dip_best_average_profit[0])
    print("========================================")
    print("BAGUETTE BEST AVERAGE PROFIT:", baguette_best_average_profit[1])
    print("PARAMS:", baguette_best_average_profit[0])
    print("========================================")
    print("UKULELE BEST AVERAGE PROFIT:", ukulele_best_average_profit[1])
    print("PARAMS:", ukulele_best_average_profit[0])
    print("========================================")
    print("PICNIC BASKET BEST AVERAGE PROFIT:", picnic_basket_best_average_profit[1])
    print("PARAMS:", picnic_basket_best_average_profit[0])


    print("Total time: ", time.time() - start_time, "s")
