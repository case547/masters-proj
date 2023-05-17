from helper_functions import linear_schedule

# Atari experiment hyperparams from Proximal Policy Optimization Algorithms
# by Schulman et al. (Table 5) See https://arxiv.org/abs/1707.06347 

schulman = {
    "learning_rate": linear_schedule(2.5e-4),
    "batch_size": 256,
    "n_epochs": 3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": linear_schedule(0.1),
    "ent_coef": 0.01,
    "vf_coef": 1,
    # "num_vec_envs": 8  # would go in env initialisation
}


# Hyperparms from RL Baselines3 Zoo - PPO for Atari (where applicable). See
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

rl_zoo3 = {
    "learning_rate": linear_schedule(2.5e-4),
    "n_steps": 256,  # originally 128, but changed for n_steps be int multiple of batch_size
    "batch_size": 256,
    "n_epochs": 4,
    # gamma=0.99 and gae_lambda=0.95 by default
    "clip_range": linear_schedule(0.1),
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    # "n_envs": 8  # would go in env initialisation
}


# IPPO hyperparams from RESCO supplement/appendix (Table 5)
# https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f0935e4cd5920aa6c7c996a5ee53a70f-Abstract-round1.html

resco = {
    "learning_rate": 2.5e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 1e-3,
    "max_grad_norm": 0.5,
}


# Custom hyperparams - based on rl_zoo3 but more suited for 1M steps rather than 10M

custom = {
    "learning_rate": 2.5e-4,
    "n_steps": 256,  # originally 128, but changed for n_steps be int multiple of batch_size
    "batch_size": 256,
    "n_epochs": 4,
    # gamma=0.99 and gae_lambda=0.95 by default
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    # "n_envs": 8  # would go in env initialisation
}