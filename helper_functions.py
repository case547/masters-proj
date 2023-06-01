from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from sumo_rl import TrafficSignal
import traci


def get_total_waiting_time(ts: Optional[TrafficSignal] = None) -> float:
    """Return the waiting time for a collection of vehicles.
    
    If `ts` is provided, this is only vehicles at the intersection,
    otherwise it is all vehicles present in the simulation.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    if ts:
        return sum(ts.sumo.lane.getWaitingTime(lane) for lane in ts.lanes)
    
    return sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())


def get_tyre_pm(ts: Optional[TrafficSignal] = None) -> float:
    """Return tyre PM emission based on absolute acceleration.
    
    Tyre PM emission and vehicle absolute acceleration are assumed to have a linear relationship.
    If `ts` is provided, only vehicles at the intersection are counted, otherwise it is all
    vehicles in the simulation.

    Keyword arguments
        ts: the TrafficSignal object
    """
    tyre_pm = 0

    if ts:
        for lane in ts.lanes:
            veh_list = ts.sumo.lane.getLastStepVehicleIDs(lane)
            
            for veh in veh_list:
                accel = ts.sumo.vehicle.getAcceleration(veh)
                tyre_pm += abs(accel)
    else:
        for veh in traci.vehicle.getIDList():
            accel = traci.vehicle.getAcceleration(veh)
            tyre_pm += abs(accel)

    return tyre_pm


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear schedule for learning rate and clipping parameter `clip_range`.

    :param initial_value:
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def pz_env(**kwargs):
    """Instantiate a PettingZoo environment using `CountAllRewardsEnvPZ`."""
    from envs import CountAllRewardsEnvPZ

    env = CountAllRewardsEnvPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

make_parallel_env = parallel_wrapper_fn(pz_env)


def sumo_vec_env(
        num_envs=1,
        seed: Optional[int] = None,
        start_index: int = 0,
        env_kwargs: Optional[Dict[str, Any]] = None,
):
    """Create a wrapped and monitored (single-agent) `CountAllRewardsEnv` using Stable_Baselines3's `DummyVecEnv`.
    
    :return: Stable Baselines3 `DummyVecEnv`
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import compat_gym_seed
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.patch_gym import _patch_env
    from envs import CountAllRewardsEnv
    
    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = CountAllRewardsEnv(**env_kwargs)
            env = Monitor(env)
            env = _patch_env(env)

            if seed is not None:
                compat_gym_seed(env, seed=seed + rank)
                env.action_space.seed(seed + rank)

            return env
        
        return _init
    
    return DummyVecEnv([make_env(i + start_index) for i in range(num_envs)])
