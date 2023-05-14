import csv
from collections import defaultdict
from typing import Union

import traci
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

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
        if isinstance(self.env, aec_to_parallel_wrapper):
            # Results same for any/all markov vector envs, because actions are deterministic(?)
            self.ts_dict = self.env.unwrapped.vec_envs[-1].par_env.unwrapped.env.traffic_signals
        else:
            self.ts_dict = self.env.get_attr("traffic_signals")[0]


    def step(self, t) -> bool:
        self.t_step += 1

        # In this time step
        stats = defaultdict(float)

        # Sum stats for traffic signals
        for ts in self.ts_dict.values():
            stats["arrived"] += traci.simulation.getArrivedNumber()
            stats["avg_speed"] += ts.get_average_speed()
            stats["pressure"] += ts.get_pressure()
            stats["queued"] += ts.get_total_queued()
            stats["tyre_pm"] += get_tyre_pm(ts)
            stats["wait_time"] += get_total_waiting_time(ts)

        # Log to CSV
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([self.step] + list(stats.values()))

        # Update counters
        self.tyre_pm_cumulative += stats["tyre_pm"]
        self.arrived_so_far += stats["arrived"]

        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("eval/arrived", self.arrived_so_far, self.t_step)
            self.tb_writer.add_scalar("eval/tyre_pm", self.tyre_pm_cumulative, self.t_step)
            tb_stats = list(stats.keys())
            tb_stats.remove("arrived")
            tb_stats.remove("tyre_pm")
            for s in tb_stats:
                self.tb_writer.add_scalar(f"eval/{s}", stats[s], self.t_step)
         
        return True
