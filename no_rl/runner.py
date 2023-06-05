import argparse
import csv
import os
import sys
from pathlib import PurePath

import numpy as np
import pandas as pd
from sumolib import checkBinary
import traci

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")


def run(net_name: str, seed: int):
    """Execute the baseline evaluation via TraCI."""
    csv_dir = os.path.join("no_rl", net_name)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # Create CSV
    metrics_csv = os.path.join(csv_dir,f"baseline_{seed}.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time", "arrived_num", "tyre_pm", "stopped", "total_wait", "avg_wait", "avg_speed"])

    listener = SimListener(csv_path=metrics_csv)
    traci.addStepListener(listener)

    end_time = traci.simulation.getEndTime()
    for _ in range(int(end_time)):
        traci.simulationStep()

    traci.close()

    # Collate results
    df = pd.read_csv(metrics_csv)
    total_arrived = sum(df["arrived_num"])
    total_tyre_pm = sum(df["tyre_pm"])
    mean_stopped = np.mean(df["stopped"])
    mean_total_wait = np.mean(df["total_wait"])
    mean_avg_wait = np.mean(df["avg_wait"])
    mean_avg_speed = np.mean(df["avg_speed"])

    collate_csv = os.path.join(csv_dir, "collated_results.csv")
    with open(collate_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seed, total_arrived, total_tyre_pm, mean_stopped,
                         mean_total_wait, mean_avg_wait, mean_avg_speed])
    

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="SUMO configuration file to run", required=True)
    parser.add_argument("-n", "--num-seeds", help="number of seeds to run simulations for", default=20)
    parser.add_argument("--gui", action="store_true", help="run the GUI version of sumo")
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    from listeners import SimListener

    options = parse_options()
    network = PurePath(options.file).parts[-2]
    num_seeds = options.num_seeds

    # When this script is called from the command line, it will start
    # sumo as a server, then connect and run the rest of the code
    if options.gui:
        sumo_binary = checkBinary("sumo-gui")
    else:
        sumo_binary = checkBinary("sumo")

    # The normal way of using traci: Sumo is started as a
    # subprocess and then the python script connects and runs
    start_seed = 23423
    for rank in range(num_seeds):
        print(f"Starting simulation with seed {start_seed+rank} ({rank+1}/{num_seeds})")

        traci.start([sumo_binary, "-c", options.file, "--seed", str(start_seed+rank)])
        run(network, start_seed + rank)
