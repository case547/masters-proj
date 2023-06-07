import argparse
import csv
import os
import sys
from pathlib import PurePath

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from sumolib import checkBinary
import traci

# # Need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

from envs import CountAllRewardsEnv
from evaluate import evaluate
from reward_functions import combined_reward


def run(net_name: str, seed: int, model_path: str):
    """Execute the SB3 environment evaluation and log to CSV."""
    set_random_seed(seed)

    env_params = {
        "net_file": os.path.join("nets", net_name, f"{net_name}.net.xml"),
        "route_file": os.path.join("nets", net_name, f"{net_name}.rou.xml"),
        "num_seconds": 3600,
        "single_agent": True,
        "reward_fn": combined_reward,
        "sumo_seed": seed,
    }
    eval_env = CountAllRewardsEnv(**env_params)
    eval_env.reset(seed=seed)
    eval_env = Monitor(eval_env)  # wrap env to know episode reward, length, time

    # Create CSV
    metrics_csv = os.path.join(net_name, f"seed_{seed}.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time", "arrived_num", "sys_tyre_pm", "sys_stopped",
                         "sys_total_wait", "sys_avg_wait", "sys_avg_speed",
                         "agents_tyre_pm", "agents_stopped", "agents_total_wait",
                         "agents_avg_speed", "agents_total_pressure"])

    # Perform evaluation
    model = PPO.load(model_path)
    evaluate(model, eval_env, metrics_csv)

    # Collate results
    df = pd.read_csv(metrics_csv)
    total_arrived = sum(df["arrived_num"])
    total_tyre_pm = sum(df["sys_tyre_pm"])
    mean_stopped = np.mean(df["sys_stopped"])
    mean_total_wait = np.mean(df["sys_total_wait"])
    mean_avg_wait = np.mean(df["sys_avg_wait"])
    mean_avg_speed = np.mean(df["sys_avg_speed"])

    collate_csv = os.path.join(net_name, "collated_results.csv")
    with open(collate_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seed, total_arrived, total_tyre_pm, mean_stopped,
                         mean_total_wait, mean_avg_wait, mean_avg_speed])
        
    eval_env.close()
    

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="folder containing a network and route file", required=True)
    parser.add_argument("-m", "--model", help="path to the saved model", required=True)
    parser.add_argument("-n", "--num-seeds", help="number of random seeds", default=20)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_options()
    network = PurePath(options.folder).parts[-1]
    num_seeds = options.num_seeds

    start_seed = 23423
    for rank in range(num_seeds):
        print(f"Starting simulation with seed {start_seed+rank} ({rank+1}/{num_seeds})")
        run(network, start_seed + rank, options.model)
