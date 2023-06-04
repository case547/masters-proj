import argparse
import csv
import os
import sys
from pathlib import PurePath

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

    # Create CSV
    csv_path = os.path.join(csv_dir,f"baseline_{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time", "arrived_num", "sys_tyre_pm", "sys_stopped",
                         "sys_total_wait", "sys_avg_wait", "sys_avg_speed",])

    listener = SimListener(csv_path=csv_path)
    traci.addStepListener(listener)

    end_time = traci.simulation.getEndTime()
    for _ in range(int(end_time)):
        traci.simulationStep()

    traci.close()


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="SUMO configuration file to run")
    parser.add_argument("--gui", action="store_true", help="run the GUI version of sumo")
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    from listeners import SimListener

    options = parse_options()
    network = PurePath(options.file).parts[-2]

    # When this script is called from the command line, it will start
    # sumo as a server, then connect and run the rest of the code
    if options.gui:
        sumo_binary = checkBinary("sumo-gui")
    else:
        sumo_binary = checkBinary("sumo")

    # The normal way of using traci: Sumo is started as a
    # subprocess and then the python script connects and runs
    start_seed = 23423
    num_seeds = 30
    for rank in range(num_seeds):
        print(f"Starting simulation with seed {start_seed+rank} ({rank+1}/{num_seeds})")

        traci.start([sumo_binary, "-c", options.file, "--seed", str(start_seed+rank)])
        run(network, start_seed + rank)
