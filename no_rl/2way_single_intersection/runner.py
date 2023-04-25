import csv
import os
import optparse
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


def run(folder, csv_title, end_time=None, delta_t=1):
    """Execute the TraCI control loop."""
    # Get traffic lights and lanes they control
    tl_ids = list(traci.trafficlight.getIDList())
    controlled_lanes = []
    for tl in tl_ids:
        controlled_lanes += list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

    # Initialise cumulative counters
    tyre_pm_cumulative = 0
    total_arrived = 0
    
    # Prep TensorBoard and CSV
    tb_writer = SummaryWriter(folder)
    with open(os.path.join(folder,f"{csv_title}_{delta_t}.csv"), "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["sim_time", "total_arrived", "tyre_pm", "tyre_pm_cumulative"])

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
        total_arrived += traci.simulation.getArrivedNumber()

        # Log to CSV
        with open(os.path.join(folder,f"{csv_title}_{delta_t}.csv"), "a", newline="") as f:
            csv_writer = csv.writer(f)

            sim_time = step * delta_t  # can also use traci.simulation.getTime()
            csv_writer.writerow([sim_time, total_arrived, tyre_pm, tyre_pm_cumulative])

        # Log to TensorBoard
        tb_writer.add_scalar("baseline/total_arrived", total_arrived, step)
        tb_writer.add_scalar("baseline/tyre_pm", tyre_pm, step)
        tb_writer.add_scalar("baseline/tyre_pm_cumulative", tyre_pm_cumulative, step)

        # Step simulation for delta_time seconds
        for _ in range(delta_t):
            traci.simulationStep()
    
    tb_writer.close()
    traci.close()
    print("Total tyre PM emitted:", tyre_pm_cumulative)
    sys.stdout.flush()


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--gui", action="store_true",
                         default=False, help="run the GUI version of sumo")
    options, _ = opt_parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()

    # When this script is called from the command line, it will start
    # sumo as a server, then connect and run the rest of the code
    if options.gui:
        sumo_binary = checkBinary("sumo-gui")
    else:
        sumo_binary = checkBinary("sumo")

    # The normal way of using traci: Sumo is started as a
    # subprocess and then the python script connects and runs
    p = PurePath(__file__)
    traci.start([sumo_binary, "-c",
                 os.path.join(p.parents[2],"nets","LucasAlegre","2way-single-intersection","single-intersection.sumocfg")])
    
    run(folder=p.parent, csv_title="results", delta_t=1)
