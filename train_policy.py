# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import argparse
import os
import sys

import gym
from stable_baselines3 import DQN, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import mujoco_envs


def parse_config():
    parser = argparse.ArgumentParser(description="Model-free RL train runner")
    parser.add_argument(
        "--env_id",
        required=True,
        choices=[
            "MJCartpole-v0",
            "MJHalfCheetah-v0",
            "MJPusher-v0",
            "MJReacher-v0",
        ],
        type=str,
    )
    parser.add_argument("--train_steps", default=int(1e6), type=int)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--policy_name", required=True, choices=["TD3"], type=str)
    parser.add_argument("--exp_name", required=False, default="default", type=str)
    parser.add_argument("--device", default="auto", type=str)
    config = parser.parse_args()
    return config


def train(config):
    env = gym.make(config.env_id)
    env = DummyVecEnv([lambda: Monitor(env) for i in range(config.n_envs)])

    policy_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": True,
        "device": config.device,
        "tensorboard_log": os.path.join("data/logs", config.env_id, config.policy_name),
    }

    if config.policy_name == "TD3":
        policy = TD3(**policy_kwargs)
    else:
        raise ValueError

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join("data", "checkpoints", config.env_id, config.policy_name, config.exp_name),
        log_path="./data/logs/",
        eval_freq=50_000,
        deterministic=True,
        render=False,
    )

    try:
        policy.learn(total_timesteps=config.train_steps, callback=eval_callback, tb_log_name=config.exp_name)
    except Exception as e:
        print("Exception during training:", e)
    finally:
        policy.save(
            path=os.path.join(
                "data",
                "checkpoints",
                config.env_id,
                config.policy_name,
                config.exp_name,
                f"{config.train_steps}_steps.zip",
            )
        )


if __name__ == "__main__":
    config = parse_config()
    print(config)
    train(config)
