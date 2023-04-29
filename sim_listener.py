import csv
import os
import sys
from collections import defaultdict

import traci
from sumo_rl import SumoEnvironment
from torch.utils.tensorboard import SummaryWriter

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

from helper_functions import get_total_waiting_time, get_tyre_pm



class SimListener(traci.StepListener):
    """Custom step listener for recording to Tensorboard and CSV."""
    def __init__(self, env: SumoEnvironment, csv_path: str, tb_log_dir: str) -> None:
        self.env = env
        self.csv_path = csv_path
        self.tb_log_dir = tb_log_dir

        # Get traffic lights and lanes they control
        tl_ids = list(traci.trafficlight.getIDList())
        self.controlled_lanes = []
        for tl in tl_ids:
            self.controlled_lanes += list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

        # Initialise cumulative counters
        self.tyre_pm_cumulative = 0.
        self.arrived_so_far = 0.

        # Prep TensorBoard
        self.tb_writer = SummaryWriter(self.tb_log_dir)


    def step(self, t) -> bool:
        sim_time = traci.simulation.getTime()

        # In this time step
        stats = defaultdict(float)

        # Sum stats over all traffic light agents
        for ts_dict in self.env.get_attr("traffic_signals"):
            for ts in ts_dict.values():
                stats["arrived"] += traci.simulation.getArrivedNumber()
                stats["avg_speed"] += ts.get_average_speed()
                stats["pressure"] += ts.get_pressure()
                stats["queued"] += ts.get_total_queued()
                stats["tyre_pm"] += get_tyre_pm(ts)
                stats["wait_time"] += get_total_waiting_time(ts)

        # Log to CSV
        with open(self.csv_path, "a", encoding="ansi", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([sim_time] + list(stats.values()))

        # Update counters
        self.tyre_pm_cumulative += stats["tyre_pm"]
        self.arrived_so_far += stats["arrived"]

        # Log to TensorBoard
        self.tb_writer.add_scalar("eval/arrived", self.arrived_so_far, sim_time)
        self.tb_writer.add_scalar("eval/tyre_pm", self.tyre_pm_cumulative, sim_time)
        tb_stats = list(stats.keys())
        tb_stats.remove("arrived")
        tb_stats.remove("tyre_pm")
        for s in tb_stats:
            self.tb_writer.add_scalar(f"eval/{s}", stats[s], sim_time)
            
        return True


    def cleanUp(self) -> None:
        self.tb_writer.close()
        sys.stdout.flush()
