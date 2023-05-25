import csv
from collections import defaultdict
from typing import Union

import traci
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

from envs import CountAllRewardsEnvPZ
from helper_functions import get_total_waiting_time, get_tyre_pm


class SimListener(traci.StepListener):
    """Custom step listener for recording to Tensorboard and CSV."""
    def __init__(self, env: Union[Monitor, VecMonitor], csv_path: str = None, tb_log_dir: str = None) -> None:
        self.env = env
        self.csv_path = csv_path

        if tb_log_dir:
            self.tb_writer = SummaryWriter(tb_log_dir)  # prep TensorBoard

        # Get traffic lights and lanes they control
        tl_ids = list(traci.trafficlight.getIDList())
        self.controlled_lanes = []
        for tl in tl_ids:
            self.controlled_lanes += list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl)))

        # Initialise cumulative counters
        self.tyre_pm_cumulative = 0.
        self.arrived_so_far = 0.

        self.t_step = 0.

        # Get traffic signal objects
        if isinstance(self.env.unwrapped.vec_envs[0].par_env.unwrapped, CountAllRewardsEnvPZ):
            self.ts_dict = self.env.unwrapped.vec_envs[0].par_env.unwrapped.env.traffic_signals
        else:
            self.ts_dict = self.env.get_attr("traffic_signals")[0]


    def step(self, t) -> bool:
        # In this time step
        stats = defaultdict(float)

        # Sum stats for traffic signals
        for ts in self.ts_dict.values():
            stats["arrived_num"] += traci.simulation.getArrivedNumber()
            stats["avg_speed"] += ts.get_average_speed()
            stats["pressure"] += ts.get_pressure()
            stats["queued"] += ts.get_total_queued()
            stats["tyre_pm"] += get_tyre_pm(ts)
            stats["waiting_time"] += get_total_waiting_time(ts)

        # Log to CSV
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([self.t_step] + list(stats.values()))

        # Update counters
        self.tyre_pm_cumulative += stats["tyre_pm"]
        self.arrived_so_far += stats["arrived_num"]

        # Log to TensorBoard
        if hasattr(self, "tb_writer"):
            self.tb_writer.add_scalar("stats/arrived_so_far", self.arrived_so_far, self.t_step)
            self.tb_writer.add_scalar("stats/tyre_pm_cumulative", self.tyre_pm_cumulative, self.t_step)
            tb_stats = list(stats.keys())
            tb_stats.remove("arrived_num")
            tb_stats.remove("tyre_pm")
            for s in tb_stats:
                self.tb_writer.add_scalar(f"stats/{s}", stats[s], self.t_step)
        
        self.t_step += 1
        return True
