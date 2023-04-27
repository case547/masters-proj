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


def run(folder, end_time=None, delta_t=1):
    """Execute the TraCI control loop."""
    # Get traffic lights and lanes they control
    tl_ids = list(traci.trafficlight.getIDList())
    controlled_lanes = []
    for tl in tl_ids:
        controlled_lanes += list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

    # Initialise cumulative counters
    tyre_pm_cumulative = 0
    arrived_so_far = 0
    
    # Prep TensorBoard and CSV
    tb_writer = SummaryWriter(folder)
    with open(os.path.join(folder,f"results_{delta_t}.csv"), "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["sim_time", "arrived", "tyre_pm"])

    if not end_time:
        end_time = traci.simulation.getEndTime()
    assert end_time % delta_t == 0
    num_steps = int(end_time / delta_t)

    for step in range(num_steps):
        tyre_pm = 0  # in this time step

        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for v in vehicles:
                accel = traci.vehicle.getAcceleration(v)
                tyre_pm += abs(accel) * delta_t

        # Update counters    
        tyre_pm_cumulative += tyre_pm
        arrived_so_far += traci.simulation.getArrivedNumber()

        # Log to CSV
        with open(os.path.join(folder,f"results_{delta_t}.csv"), "a", newline="") as f:
            csv_writer = csv.writer(f)

            sim_time = step * delta_t  # can also use traci.simulation.getTime()
            csv_writer.writerow([sim_time, arrived_so_far, tyre_pm, tyre_pm_cumulative])

        # Log to TensorBoard
        tb_writer.add_scalar("baseline/arrived", arrived_so_far, step)
        tb_writer.add_scalar("baseline/tyre_pm", tyre_pm_cumulative, step)

        # Step simulation for delta_time seconds
        for _ in range(delta_t):
            traci.simulationStep()
    
    tb_writer.close()
    traci.close()
    print("Total tyre PM emitted:", tyre_pm_cumulative)
    sys.stdout.flush()


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help=".sumocfg file to run")
    parser.add_argument("--dt", default=1, type=int, help=".sumocfg file to run")
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
    run(folder=os.path.join("no_rl", network), delta_t=options.dt)
