import argparse
import csv
import os
import sys
from pathlib import PurePath

import traci
from sumolib import checkBinary
from torch.utils.tensorboard import SummaryWriter

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary


def run(net_name):
    """Execute the baseline evaluation via TraCI."""
    # Get traffic lights and lanes they control
    tl_ids = list(traci.trafficlight.getIDList())
    controlled_lanes = []
    for tl in tl_ids:
        controlled_lanes += list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

    # Initialise cumulative counters
    tyre_pm_cumulative = 0
    arrived_so_far = 0

    last_wait_time = 0
    
    # Prep TensorBoard and CSV
    tb_writer = SummaryWriter(os.path.join("logs",net_name,"baseline"))
    with open(os.path.join("no_rl",f"{net_name}.csv"), "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["sim_time", "arrived", "tyre_pm", "wait_time", "delta_wait_time"])

    end_time = traci.simulation.getEndTime()
    
    for step in range(int(end_time)):
        traci.simulationStep()

        # In this time step
        arrived_num = traci.simulation.getArrivedNumber()
        tyre_pm = 0
        wait_time = 0

        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for v in vehicles:
                accel = traci.vehicle.getAcceleration(v)
                tyre_pm += abs(accel)

            wait_time += traci.lane.getWaitingTime(lane)

        diff_wait_time = wait_time - last_wait_time
        last_wait_time = wait_time

        # Log to CSV
        with open(os.path.join("no_rl",f"{net_name}.csv"), "a", newline="") as f:
            csv_writer = csv.writer(f)

            sim_time = step  # can also use traci.simulation.getTime()
            csv_writer.writerow([sim_time, arrived_num, tyre_pm, wait_time, diff_wait_time])

        # Update counters
        tyre_pm_cumulative += tyre_pm
        arrived_so_far += arrived_num

        # Log to TensorBoard
        tb_writer.add_scalar("eval/arrived_so_far", arrived_so_far, step)
        tb_writer.add_scalar("eval/tyre_pm_cumulative", tyre_pm_cumulative, step)
    
    tb_writer.close()
    traci.close()
    print("Vehicles arrived:", arrived_so_far)
    print("Tyre PM emitted:", tyre_pm_cumulative)
    print("Final wait time:", last_wait_time)
    sys.stdout.flush()

    with open(os.path.join("no_rl","baseline_results.csv"), "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([net_name, arrived_so_far, tyre_pm_cumulative, last_wait_time])


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="SUMO configuration file to run")
    parser.add_argument("--gui", action="store_true", help="run the GUI version of sumo")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_options()

    # When this script is called from the command line, it will start
    # sumo as a server, then connect and run the rest of the code
    if options.gui:
        sumo_binary = checkBinary("sumo-gui")
    else:
        sumo_binary = checkBinary("sumo")

    # The normal way of using traci: Sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumo_binary, "-c", options.file])
    
    network = PurePath(options.file).parts[-2]
    run(network)
