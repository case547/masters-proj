import argparse
import csv
import os
import random
from typing import Optional
from pathlib import PurePath

import numpy as np
import pandas as pd
import torch

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.tune.registry import register_env

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

from envs import MultiAgentSumoEnv
from observation import Grid2x2ObservationFunction
from reward_functions import combined_reward

OBS_CLASS = Grid2x2ObservationFunction

def env_creator(env_name, begin_time, seed: Optional[int] = None, csv_path: Optional[str] = None):
    env_params = {
        "net_file": os.path.join("nets", env_name, f"{env_name}.net.xml"),
        "route_file": os.path.join("nets", env_name, f"{env_name}.rou.xml"),
        "begin_time": begin_time,
        "num_seconds": 3600,
        "reward_fn": combined_reward,
        "sumo_seed": seed,
        "observation_class": OBS_CLASS,
        "add_system_info": False,
    }
    env = MultiAgentSumoEnv(eval=True, csv_path=csv_path, **env_params)
    return env

def run(net_name: str, seed: int, checkpoint_path: str, begin_time: int):
    """Execute the SB3 environment evaluation and log to CSV."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MultiAgentEnvCompatibility(env_creator(net_name, begin_time, seed=seed))

    config: PPOConfig
    config = (
        PPOConfig()
        .environment(env=net_name)
        .framework(framework="torch")
        .rollouts(
            rollout_fragment_length=100,
            num_rollout_workers=10,
        )
        .training(
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            train_batch_size=1000,
            sgd_minibatch_size=100,
            num_sgd_iter=10,
        )
        .evaluation(
            evaluation_duration=1,
            evaluation_num_workers=1,
            evaluation_sample_timeout_s=300,
        )
        .debugging(seed=seed)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        .multi_agent(
            policies=set(env.env.ts_ids),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .fault_tolerance(recreate_failed_workers=True)
    )

    # Create CSV
    if not os.path.exists(net_name):
        os.makedirs(net_name)

    metrics_csv = os.path.join(net_name, f"seed_{seed}.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time", "arrived_num", "sys_tyre_pm", "sys_stopped",
                         "sys_total_wait", "sys_avg_wait", "sys_avg_speed",
                         "agents_tyre_pm", "agents_stopped", "agents_total_wait",
                         "agents_avg_speed", "agents_total_pressure"])

    register_env(net_name, lambda config: MultiAgentEnvCompatibility(
        env_creator(net_name, begin_time, seed=seed, csv_path=metrics_csv)
    ))
    ray.init()

    checkpoint_path = os.path.abspath(checkpoint_path)
    ppo_agent = PPO.from_checkpoint(checkpoint_path)

    ppo_agent.evaluate()

    ray.shutdown()

    # Collate results
    df = pd.read_csv(metrics_csv)
    total_arrived = sum(df["arrived_num"][:3600])
    total_tyre_pm = sum(df["sys_tyre_pm"][:3600])
    mean_stopped = np.mean(df["sys_stopped"][:3600])
    mean_total_wait = np.mean(df["sys_total_wait"][:3600])
    mean_avg_wait = np.mean(df["sys_avg_wait"][:3600])
    mean_avg_speed = np.mean(df["sys_avg_speed"][:3600])

    collate_csv = os.path.join(net_name, "collated_results.csv")
    with open(collate_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seed, total_arrived, total_tyre_pm, mean_stopped,
                         mean_total_wait, mean_avg_wait, mean_avg_speed])
        
    env.close()
    

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="folder containing a network and route file", required=True)
    parser.add_argument("-m", "--model", help="path to the saved model", required=True)
    parser.add_argument("-n", "--num-seeds", help="number of random seeds", default=20)
    parser.add_argument("--begin-time", help="number of random seeds", default=0)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_options()
    network = PurePath(options.folder).parts[-1]
    num_seeds = options.num_seeds
    begin_time = options.begin_time

    start_seed = 23423
    for rank in range(num_seeds):
        print(f"Starting simulation with seed {start_seed+rank} ({rank+1}/{num_seeds})")
        run(network, start_seed + rank, options.model, begin_time)
