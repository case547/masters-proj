from collections import Counter
import csv
import os
import time
from typing import Optional, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from gymnasium.utils import EzPickle
from pettingzoo.utils import agent_selector
from sumo_rl import SumoEnvironment
from sumo_rl.environment.env import SumoEnvironmentPZ
import traci

from helper_functions import get_total_waiting_time, get_tyre_pm


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class CountAllRewardsEnv(SumoEnvironment):
    """Environment that counts rewards every sumo_step.
    
    Because delta_time != 1, the reward given to the agent(s) every
    step() is the sum of the last delta_time rewards generated by SUMO.
    """
    def __init__(self, **kwargs):
        # Call the parent constructor
        super().__init__(**kwargs)
    
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            # Rewards for the sumo steps between every env step
            self.reward_hold = Counter({ts: 0 for ts in self.ts_ids})

            for _ in range(self.delta_time):
                self._sumo_step()

                r = {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}
                self.reward_hold.update(r)  # add r to reward_hold Counter

        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info
    
    def _run_steps(self):
        # Rewards for the sumo steps between every env step
        self.reward_hold = Counter({ts: 0 for ts in self.ts_ids})
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            r = {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}
            self.reward_hold.update(r)  # add r to reward_hold Counter

            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _compute_rewards(self):
        self.rewards.update(
            {ts: self.reward_hold[ts] for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}


class CountAllRewardsEnvPZ(SumoEnvironmentPZ):
    """A wrapper for `CountAllRewardsEnv` that implements the AECEnv interface from PettingZoo."""
    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = CountAllRewardsEnv(**self._kwargs)  # instead of SumoEnvironment

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}


class MultiAgentSumoEnv(CountAllRewardsEnv):
    def __init__(self, eval=False, csv_path: Optional[str] = None, tb_log_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.eval = eval
        self.csv_path = csv_path
        self.tb_log_dir = tb_log_dir

        if tb_log_dir:
            self.tb_writer = SummaryWriter(tb_log_dir)  # prep TensorBoard

        # Initialise cumulative counters
        self.tyre_pm_system = 0
        self.tyre_pm_agents = 0
        self.arrived_so_far = 0

    def _compute_info(self):
        info = {"__common__": {"step": self.sim_step}}
        per_agent_info = self._get_per_agent_info()

        for agent_id in self.ts_ids:
            agent_info = {}

            for k, v in per_agent_info.items():
                if k.startswith(agent_id):
                    agent_info[k.split("_")[-1]] = v

            # Add tyre PM
            agent_info["tyre_pm"] = get_tyre_pm(self.traffic_signals[agent_id])

            info.update({agent_id: agent_info})

        return info

    def _sumo_step(self):
        self.sumo.simulationStep()

        if self.eval:
            # Get system stats
            vehicles = self.sumo.vehicle.getIDList()
            speeds = [self.sumo.vehicle.getSpeed(veh) for veh in vehicles]
            waiting_times = [self.sumo.vehicle.getWaitingTime(veh) for veh in vehicles]

            system_tyre_pm = get_tyre_pm()
            arrived_num = self.sumo.simulation.getArrivedNumber()
            self.tyre_pm_system += system_tyre_pm
            self.arrived_so_far += arrived_num
            
            system_stats = {
                "total_stopped": sum(int(speed < 0.1) for speed in speeds),
                "total_waiting_time": sum(waiting_times),
                "mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
                "mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            }
            
            # Get agent stats
            agents_tyre_pm = sum(get_tyre_pm(ts) for ts in self.traffic_signals.values())
            self.tyre_pm_agents += agents_tyre_pm
            
            agent_stats = {
                # ts: TrafficSignal
                "total_stopped": sum(ts.get_total_queued() for ts in self.traffic_signals.values()),
                "total_waiting_time": sum(get_total_waiting_time(ts) for ts in self.traffic_signals.values()),
                "average_speed": np.mean([ts.get_average_speed() for ts in self.traffic_signals.values()]),
                "total_pressure": sum(-ts.get_pressure() for ts in self.traffic_signals.values())
            }
            
            # Log to CSV
            if self.csv_path:
                with open(self.csv_path, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    data = ([self.sim_step, arrived_num, system_tyre_pm]
                            + list(system_stats.values())
                            + [agents_tyre_pm]
                            + list(agent_stats.values()))
                    
                    csv_writer.writerow(data)

            # Log to TensorBoard
            if hasattr(self, "tb_writer"):
                # System
                self.tb_writer.add_scalar("world/arrived_so_far", self.arrived_so_far, self.sim_step)
                self.tb_writer.add_scalar("world/tyre_pm_cumulative", self.tyre_pm_system, self.sim_step)

                for stat, val in system_stats.items():
                    self.tb_writer.add_scalar(f"world/{stat}", val, self.sim_step)

                # Agents
                self.tb_writer.add_scalar("agents/tyre_pm_cumulative", self.tyre_pm_agents, self.sim_step)

                for stat, val in agent_stats.items():
                    self.tb_writer.add_scalar(f"agents/{stat}", val, self.sim_step)

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        # Help completely release SUMO port between episodes to address
        # "Unable to create listening socket: Address already in use" error
        time.sleep(5)

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None