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


def run(folder, csv_path):
    """execute the TraCI control loop"""
    # Get traffic lights and lanes they control
    tl_ids = list(traci.trafficlight.getIDList())
    controlled_lanes = []
    for tl in tl_ids:
        controlled_lanes += list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

    tyre_pm_cumulative = 0.  # accumulating across time steps
    step = 0
    
    # Prep TensorBoard and CSV
    tb_writer = SummaryWriter(folder)
    with open(csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["sim_time","not_arrived","tyre_pm","accum_tyre_pm"])

    MAX_TIME = 1e5

    for step in range(MAX_TIME):
        tyre_pm = 0.  # in this time step

        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
           
            for v in vehicles:
                accel = traci.vehicle.getAcceleration(v)
                tyre_pm += abs(accel)
            
        tyre_pm_cumulative += tyre_pm
        
        sim_time = traci.simulation.getTime()
        not_arrived = traci.simulation.getMinExpectedNumber()

        # Log to CSV
        with open(os.path.join(folder,"2way_single_intersection.csv"), "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([sim_time, not_arrived, tyre_pm, tyre_pm_cumulative])

        # Log to TensorBoard
        tb_writer.add_scalar("not_arrived", not_arrived, step)
        tb_writer.add_scalar("tyre_pm", tyre_pm, step)
        tb_writer.add_scalar("tyre_pm_cumulative", tyre_pm_cumulative, step)
        
        traci.simulationStep()
    
    tb_writer.close()
    traci.close()
    print("Total tyre PM emitted:", tyre_pm_cumulative)
    sys.stdout.flush()


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the command line version of sumo")
    options, _ = opt_parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()

    # When this script is called from the command line, it will start
    # sumo as a server, then connect and run the rest of the code
    if options.nogui:
        sumo_binary = checkBinary("sumo")
    else:
        sumo_binary = checkBinary("sumo-gui")

    # The normal way of using traci: Sumo is started as a
    # subprocess and then the python script connects and runs
    p = PurePath(__file__)
    traci.start([sumo_binary, "-c",
                 os.path.join(p.parents[2],"nets","2way-single-intersection","single-intersection.sumocfg")])
    
    run(folder=p.parent, csv_path="results.csv")
