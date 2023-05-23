from collections import Counter
from typing import Union

import gymnasium
from sumo_rl import SumoEnvironment


class AccumRewardsWrapper(gymnasium.Wrapper):
    """
    Args:
        env: (SumoEnvironment) SUMO environment that will be wrapped
    """

    def __init__(self, env: SumoEnvironment):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
    
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            # Rewards for the sumo steps between every env step
            self.reward_hold = Counter({ts: 0 for ts in self.env.ts_ids})

            for _ in range(self.env.delta_time):
                self.env._sumo_step()

                r = {ts: self.env.traffic_signals[ts].compute_reward() for ts in self.env.ts_ids}
                self.reward_hold.update(r)  # add r to reward_hold Counter

        else:
            self.env._apply_actions(action)
            self._run_steps()

        observations = self.env._compute_observations()
        rewards = self._compute_rewards()
        dones = self.env._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self.env._compute_info()

        if self.env.single_agent:
            return observations[self.env.ts_ids[0]], rewards[self.env.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info
    
    def _run_steps(self):
        # Rewards for the sumo steps between every env step
        self.reward_hold = Counter({ts: 0 for ts in self.env.ts_ids})
        time_to_act = False
        while not time_to_act:
            self.env._sumo_step()
            r = {ts: self.env.traffic_signals[ts].compute_reward() for ts in self.env.ts_ids}
            self.reward_hold.update(r)  # add r to reward_hold Counter

            for ts in self.env.ts_ids:
                self.env.traffic_signals[ts].update()
                if self.env.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _compute_rewards(self):
        self.env.rewards.update(
            {ts: self.reward_hold[ts] for ts in self.env.ts_ids if self.env.traffic_signals[ts].time_to_act}
        )
        return {ts: self.env.rewards[ts] for ts in self.env.rewards.keys() if self.env.traffic_signals[ts].time_to_act}
