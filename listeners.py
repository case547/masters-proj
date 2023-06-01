import csv
from typing import Dict, Optional

import numpy as np
from sumo_rl import TrafficSignal
from torch.utils.tensorboard import SummaryWriter
import traci

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

from helper_functions import get_total_waiting_time, get_tyre_pm


class SimListener(traci.StepListener):
    """Custom step listener for recording to Tensorboard and CSV."""
    def __init__(self, env=None, csv_path: Optional[str] = None, tb_log_dir: Optional[str] = None) -> None:
        self.env = env
        self.csv_path = csv_path
        if env:
            self.ts_dict = self.get_traffic_signals()

        if tb_log_dir:
            self.tb_writer = SummaryWriter(tb_log_dir)  # prep TensorBoard

        # Initialise cumulative counters
        self.tyre_pm_system = 0
        self.tyre_pm_agents = 0
        self.arrived_so_far = 0

        self.t_step = 0

    def get_traffic_signals(self) -> Dict[str, TrafficSignal]:
        """Get traffic signal objects"""
        ts_dict = self.env.unwrapped.env.traffic_signals
        return ts_dict

    def step(self, t) -> bool:
        # Get system stats
        vehicles = traci.vehicle.getIDList()
        speeds = [traci.vehicle.getSpeed(veh) for veh in vehicles]
        waiting_times = [traci.vehicle.getWaitingTime(veh) for veh in vehicles]

        system_tyre_pm = get_tyre_pm()
        arrived_num = traci.simulation.getArrivedNumber()
        self.tyre_pm_system += system_tyre_pm
        self.arrived_so_far += arrived_num
        
        system_stats = {
            "total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "total_waiting_time": sum(waiting_times),
            "mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }
        
        # Get agent stats
        if hasattr(self, "ts_dict"):
            agents_tyre_pm = sum(get_tyre_pm(ts) for ts in self.ts_dict.values())
            self.tyre_pm_agents += agents_tyre_pm
            
            agent_stats = {
                "total_stopped": sum(ts.get_total_queued() for ts in self.ts_dict.values()),
                "total_waiting_time": sum(get_total_waiting_time(ts) for ts in self.ts_dict.values()),
                "average_speed": np.mean(ts.get_average_speed() for ts in self.ts_dict.values()),
                "total_pressure": sum(-ts.get_pressure() for ts in self.ts_dict.values())
            }
        
        # Log to CSV
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                csv_writer = csv.writer(f)
                data = [traci.simulation.getTime(), arrived_num, system_tyre_pm] + list(system_stats.values())

                if hasattr(self, "ts_dict"):
                    data += [agents_tyre_pm] + list(agent_stats.values())
                
                csv_writer.writerow(data)

        # Log to TensorBoard
        if hasattr(self, "tb_writer"):
            # System
            self.tb_writer.add_scalar("world/arrived_so_far", self.arrived_so_far, self.t_step)
            self.tb_writer.add_scalar("world/tyre_pm_cumulative", self.tyre_pm_system, self.t_step)

            for stat, val in system_stats.items():
                self.tb_writer.add_scalar(f"world/{stat}", val, self.t_step)

            # Agents
            if hasattr(self, "ts_dict"):
                self.tb_writer.add_scalar("agents/tyre_pm_cumulative", self.tyre_pm_agents, self.t_step)

                for stat, val in agent_stats.items():
                    self.tb_writer.add_scalar(f"agents/{stat}", val, self.t_step)
        
        self.t_step += 1
        return True



class SB3Listener(SimListener):
    """Implementation of `SimListener` for Stable Baselines3."""

    def __init__(self, env, csv_path: Optional[str] = None, tb_log_dir: Optional[str] = None) -> None:
        super().__init__(env, csv_path, tb_log_dir)

    def get_traffic_signals(self) -> Dict[str, TrafficSignal]:
        """Get traffic signal objects"""
        from stable_baselines3.common.vec_env import VecMonitor
        
        if isinstance(self.env, VecMonitor):
            # ts_dict = self.env.unwrapped.par_env.unwrapped.env.traffic_signals
            ts_dict = self.env.unwrapped.vec_envs[0].par_env.unwrapped.env.traffic_signals
        else:
            ts_dict = self.env.get_attr("traffic_signals")[0]

        return ts_dict
