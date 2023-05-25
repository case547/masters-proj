from collections import Counter
from typing import Union

import gymnasium
from gymnasium.utils import EzPickle, seeding
from pettingzoo.utils import agent_selector, wrappers
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from sumo_rl import SumoEnvironment
from sumo_rl.environment.env import SumoEnvironmentPZ


class CountAllRewardsEnv(SumoEnvironment):
    """
    Args:
        env: (SumoEnvironment) SUMO environment that will be wrapped
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
        info = self.compute_info()

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

    # def _apply_actions(self, actions):
    #     self.env._apply_actions(actions)

    # def _compute_dones(self):
    #     self.env._compute_dones()

    # def _compute_info(self):
    #     self.env._compute_info()

    # def _compute_observations(self):
    #     self.env._compute_observations()

    # def _compute_rewards(self):
    #     self.env._compute_rewards()

    # def _sumo_step(self):
    #     self.env._sumo_step()


class CountAllRewardsEnvPZ(SumoEnvironmentPZ):
    """A wrapper for `CountAllRewardsEnv` that implements the AECEnv interface from PettingZoo."""
    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = CountAllRewardsEnv(**self._kwargs)

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