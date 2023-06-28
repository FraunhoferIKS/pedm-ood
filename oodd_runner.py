# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import argparse
import os
import pprint
from datetime import datetime
from typing import List, Optional, Union

import gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from ood_baselines import *  # noqa
from ood_baselines.common.base_detector import Base_Detector
from pedm.pedm_detector import PEDM_Detector
from utils.data import (
    load_object,
    n_policy_rollouts,
    policy_rollout,
    save_object,
    split_train_test_episodes,
)
from utils.env_utils import make_env
from utils.gpu_utils import get_device
from utils.load_utils import load_policy
from utils.save_utils import save_results
from utils.stats import eval_metrics

DEFAULT_MODS = [
    "bm_factor_minor",
    "act_factor_minor",
    "act_offset_minor",
    "act_noise_minor",
    "force_vector_minor",
    "bm_factor_severe",
    "act_factor_severe",
    "act_offset_severe",
    "act_noise_severe",
    "force_vector_severe",
]


def print_tabs(values):
    print("".join([f"{v:<20}" for v in values]))


def dict_to_str(dic):
    return "_".join([f"{k}_{v}" for k, v in dic.items()])


def train_detector(
    env: gym.Env,
    policy: BaseAlgorithm,
    data_path: str,
    n_train_episodes: int,
    detector_cls: type,
    detector_kwargs: Optional[dict] = {},
    detector_fit_kwargs: Optional[dict] = {},
):
    """
    main function to train an ood detector with data from some policy in some env

    Args:
        env: env to collect experience in
        policy: policy to interact with the env
        data_path: path to save to or load experience buffer from (if applicable)
        n_train_episodes: how many episodes to collect/train
        detector_name: type/class of the detector
        detector_path: path to save to or load detector from (if applicable)
        detector_kwargs: kwargs to pass for the detector constructor
        detector_fit_kwargs="kwargs for the training loop of the detector"

    Returns:
        detector: the trained ood detector
    """

    if os.path.exists(cfg.data_path):
        print("loading data from    :", cfg.data_path)
        ep_data = load_object(cfg.data_path)

    else:
        print(f"generating data with {policy.__class__.__name__} policy")
        ep_data = n_policy_rollouts(env=env, policy=policy, n_episodes=n_train_episodes)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        save_object(ep_data, data_path)

    train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)

    detector = detector_cls(**detector_kwargs, env=env, normalize_data=True)
    detector.fit(train_ep_data=train_ep_data, val_ep_data=val_ep_data, **detector_fit_kwargs)
    return detector


def test_detector(
    env_id: str,
    policy: BaseAlgorithm,
    ood_detector: Union[Base_Detector, PEDM_Detector],
    test_episodes: int,
    anomaly_delay: Union[str, int],
    mods: Optional[List[str]] = [None],
):
    """
    main function to test a trained detecotr on an anomalous env for several episodes.

    Args:
        env_id: name of the env
        policy: policy to interact with the env. The policy is required to have a <predict(obs)> function
                that yields a valid action
        ood_detector: detector model. The model is required to have a <predict_scores(obs, acts)> function
                that yields anomaly scores
        test_episodes: number of test episodes
        anomaly_delay: when to insert the anomaly. If None -> random time
        mods: list of mods to run on.

    Returns:
        results_dict: dictionary of all results
    """

    results_dict = {}

    print("-" * 20)
    for mod in mods:
        ep_rewards = []
        y_scores = []
        y_true = []

        for seed in range(1, test_episodes + 1):
            print(f"eval episode {seed}/{test_episodes}", flush=True, end="\r")
            env = make_env(seed=seed, env_id=env_id, anomaly_delay=anomaly_delay, mod=mod)
            obs, acts, rewards, dones = policy_rollout(env=env, policy=policy)
            ep_rewards.append(np.sum(rewards))

            if mod is not None:
                anom_scores = ood_detector.predict_scores(obs, acts)
                if env.injection_time >= len(anom_scores):
                    print("E1")
                    continue
                anom_occurrence = [0 if i < env.injection_time else 1 for i in range(len(anom_scores))]
                y_scores.extend(anom_scores)
                y_true.extend(anom_occurrence)

        auroc, ap, fpr95 = eval_metrics(y_scores, y_true)
        results_dict[mod] = {
            "mod": str(mod),
            "reward": round(np.mean(ep_rewards), 2),
            "auroc": round(auroc, 2),
            "fpr95": round(fpr95, 2),
            "ap": round(ap, 2),
        }

        if len(results_dict) == 1:
            print_tabs(results_dict[mod].keys())
        print_tabs(results_dict[mod].values())

    print("-" * 20)

    avgs_minor = {
        k: round(np.mean([results_dict[m][k] for m in mods if "minor" in m]), 2)
        for k in ["reward", "auroc", "fpr95", "ap"]
    }
    avgs_severe = {
        k: round(np.mean([results_dict[m][k] for m in mods if "severe" in m]), 2)
        for k in ["reward", "auroc", "fpr95", "ap"]
    }
    results_dict["avgs_minor"] = {"mod": "avgs_minor", **avgs_minor}
    results_dict["avgs_severe"] = {"mod": "avgs_severe", **avgs_severe}
    print(results_dict["avgs_minor"])
    print(results_dict["avgs_severe"])
    return results_dict


def run(cfg):
    pprint.pprint(cfg.__dict__)

    env = make_env(cfg.env_id)

    policy = load_policy(policy_name=cfg.policy_name, env=env, device=get_device(cfg.device), path=cfg.policy_path)

    try:
        detector_cls = globals()[cfg.detector_name]
    except Exception as e:
        raise ValueError(f"class of detector < {cfg.detector_name} > not found;", e)

    # FIXME: fix saving
    if os.path.exists(cfg.detector_path):
        ood_detector = detector_cls.load(cfg.detector_path)
    else:
        ood_detector = train_detector(
            env=env,
            policy=policy,
            data_path=cfg.data_path,
            n_train_episodes=cfg.n_train_episodes,
            detector_cls=detector_cls,
            detector_kwargs=cfg.detector_kwargs,
            detector_fit_kwargs=cfg.detector_fit_kwargs,
        )
        ood_detector.save(cfg.detector_path)

    results_dict = test_detector(
        env_id=cfg.env_id,
        policy=policy,
        ood_detector=ood_detector,
        test_episodes=cfg.test_episodes,
        anomaly_delay="random",
        mods=cfg.mods,
    )
    save_results(cfg=cfg, results_dict=results_dict)
    return results_dict


def parse_cfg(*args):
    parser = argparse.ArgumentParser(
        description="train ood_detector on nominal environments and test it on disturbed environments"
    )
    parser.add_argument(
        "--env_id",
        required=True,
        choices=[
            "MJCartpole-v0",
            "MJHalfCheetah-v0",
            "MJReacher-v0",
            "MJPusher-v0",
        ],
        type=str,
        help="which env to run on",
    )

    parser.add_argument(
        "--n_train_episodes",
        default=50,
        type=int,
        help="number of training episodes to use for training the detector (if applicable)",
    )
    parser.add_argument(
        "--test_episodes", default=100, type=int, help="number of evaluation episodes to test the detectr"
    )
    parser.add_argument(
        "--policy_name",
        default="TD3",
        choices=["TD3"],
        type=str,
        help="name/class of the policy that interacts with the env",
    )
    parser.add_argument("--policy_path", default=None, type=str, help="path to the policy file")
    parser.add_argument(
        "--mods",
        default=DEFAULT_MODS,
        type=str,
        help="which mods to evaluate on, e.g. type: << --mods \"['act_factor_severe']\" >>. if not provided, will run all mods ",
    )
    parser.add_argument(
        "--data_path",
        required=False,
        type=str,
        help="path to the databuffer if existing, if None will look at default location",
    )
    parser.add_argument("--data_tag", default="default_data", type=str, help="tag for identifying the databuffer")
    parser.add_argument("--detector_name", required=True, type=str, help="class/type of the detector to use")
    parser.add_argument("--detector_kwargs", default="{}", type=str, help="kwargs for the constructor of the detector")
    parser.add_argument("--detector_tag", default="default", type=str, help="tag for identifying the detector")
    parser.add_argument(
        "--detector_path", required=False, type=str, help="path to the model of the detector (if applicable)"
    )
    parser.add_argument(
        "--detector_fit_kwargs", default="{}", type=str, help="kwargs for the training loop of the detector"
    )
    parser.add_argument("--results_save_dir", default="data/results", type=str, help="where to save all results")
    parser.add_argument("--experiment_tag", default="testrun", type=str, help="tag for identifying the experiment")
    parser.add_argument("--device", default="auto", type=str, help="which device to use, cuda recommended!")
    cfg = parser.parse_args(*args)
    cfg.time = datetime.now().strftime("%Y%m%d_%H:%M")

    if isinstance(cfg.mods, str):
        cfg.mods = eval(cfg.mods)

    cfg.detector_kwargs = eval(cfg.detector_kwargs)
    cfg.detector_fit_kwargs = eval(cfg.detector_fit_kwargs)

    if not cfg.detector_path:
        cfg.detector_path = os.path.join(
            "data",
            "detector_models",
            cfg.env_id,
            cfg.policy_name + "_policy",
            f"{cfg.data_tag}_{cfg.n_train_episodes}_ep",
            cfg.detector_name,
            cfg.detector_tag,
            "model.pth",
        )

    if not cfg.data_path:
        cfg.data_path = os.path.join(
            "data",
            "episode_buffers_X",
            cfg.env_id,
            cfg.policy_name + "_policy",
            f"{cfg.data_tag}_{cfg.n_train_episodes}_ep",
            "ep_data.pkl",
        )

    cfg.results_save_dir = os.path.join(
        cfg.results_save_dir,
        cfg.experiment_tag,
        cfg.env_id,
        cfg.detector_name,
        cfg.detector_tag,
    )

    return cfg


if __name__ == "__main__":
    cfg = parse_cfg()
    run(cfg)
