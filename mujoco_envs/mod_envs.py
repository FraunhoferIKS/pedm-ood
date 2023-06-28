# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np

from .anom_mj_env import AnomMJEnv
from .base_envs.cartpole import CartpoleEnv
from .base_envs.half_cheetah import HalfCheetahEnv
from .base_envs.pusher import PusherEnv
from .base_envs.reacher import Reacher3DEnv


class AnomCartpoleEnv(AnomMJEnv, CartpoleEnv):
    pass


class ModCartpoleEnv(AnomCartpoleEnv):
    def __init__(self, mod, anomaly_delay):
        self.ep_length = 200
        kwargs = {}
        if mod == 1 or mod == "bm_factor_minor":
            kwargs["bm_factor"] = 1.01
        elif mod == 2 or mod == "bm_factor_severe":
            kwargs["bm_factor"] = 1.5
        elif mod == 3 or mod == "act_factor_minor":
            kwargs["act_factor"] = 1.01
        elif mod == 4 or mod == "act_factor_severe":
            kwargs["act_factor"] = 1.5
        elif mod == 5 or mod == "act_offset_minor":
            kwargs["act_offset"] = 0.01
        elif mod == 6 or mod == "act_offset_severe":
            kwargs["act_offset"] = 0.2
        elif mod == 7 or mod == "act_noise_minor":
            kwargs["act_noise"] = 0.01
        elif mod == 8 or mod == "act_noise_severe":
            kwargs["act_noise"] = 0.2
        elif mod == 9 or mod == "force_vector_minor":
            kwargs["force_vec_dict"] = {"pole": [-0.1, 0, 0, 0, 0, 0]}
        elif mod == 10 or mod == "force_vector_severe":
            kwargs["force_vec_dict"] = {"pole": [-10, 0, 0, 0, 0, 0]}
        else:
            raise ValueError("mod not speciified")
        if anomaly_delay == "random":
            anomaly_delay = np.random.randint(2, self.ep_length)
        super().__init__(injection_time=anomaly_delay, **kwargs)


class AnomHalfCheetahEnv(AnomMJEnv, HalfCheetahEnv):
    pass


class ModHalfCheetahEnv(AnomHalfCheetahEnv):
    def __init__(self, mod, anomaly_delay):
        self.ep_length = 1000
        kwargs = {}
        if mod == 1 or mod == "bm_factor_minor":
            kwargs["bm_factor"] = 1.01
        elif mod == 2 or mod == "bm_factor_severe":
            kwargs["bm_factor"] = 1.5
        elif mod == 3 or mod == "act_factor_minor":
            kwargs["act_factor"] = 1.01
        elif mod == 4 or mod == "act_factor_severe":
            kwargs["act_factor"] = 1.5
        elif mod == 5 or mod == "act_offset_minor":
            kwargs["act_offset"] = 0.01
        elif mod == 6 or mod == "act_offset_severe":
            kwargs["act_offset"] = 0.2
        elif mod == 7 or mod == "act_noise_minor":
            kwargs["act_noise"] = 0.01
        elif mod == 8 or mod == "act_noise_severe":
            kwargs["act_noise"] = 0.2
        elif mod == 9 or mod == "force_vector_minor":
            kwargs["force_vec_dict"] = {"ffoot": [-0.1, 0, 0, 0, 0, 0]}
        elif mod == 10 or mod == "force_vector_severe":
            kwargs["force_vec_dict"] = {"ffoot": [-15, 0, 0, 0, 0, 0]}
        else:
            raise ValueError("mod not speciified")
        if anomaly_delay == "random":
            anomaly_delay = np.random.randint(2, self.ep_length)
        super().__init__(injection_time=anomaly_delay, **kwargs)


class AnomPusherEnv(AnomMJEnv, PusherEnv):
    pass


class ModPusherEnv(AnomMJEnv, PusherEnv):
    def __init__(self, mod, anomaly_delay):
        self.ep_length = 150
        kwargs = {}
        if mod == 1 or mod == "bm_factor_minor":
            kwargs["bm_factor"] = 1.01
        elif mod == 2 or mod == "bm_factor_severe":
            kwargs["bm_factor"] = 2
        elif mod == 3 or mod == "act_factor_minor":
            kwargs["act_factor"] = 1.01
        elif mod == 4 or mod == "act_factor_severe":
            kwargs["act_factor"] = 2
        elif mod == 5 or mod == "act_offset_minor":
            kwargs["act_offset"] = 0.01
        elif mod == 6 or mod == "act_offset_severe":
            kwargs["act_offset"] = 0.3
        elif mod == 7 or mod == "act_noise_minor":
            kwargs["act_noise"] = 0.01
        elif mod == 8 or mod == "act_noise_severe":
            kwargs["act_noise"] = 0.3
        elif mod == 9 or mod == "force_vector_minor":
            kwargs["force_vec_dict"] = {"r_forearm_link": [-0.5, 0, 0, 0, 0, 0]}
        elif mod == 10 or mod == "force_vector_severe":
            kwargs["force_vec_dict"] = {"r_forearm_link": [-1.5, 0, 0, 0, 0, 0]}
        else:
            raise ValueError("mod not speciified")
        if anomaly_delay == "random":
            anomaly_delay = np.random.randint(2, self.ep_length)
        super().__init__(injection_time=anomaly_delay, **kwargs)


class AnomReacher3DEnv(AnomMJEnv, Reacher3DEnv):
    pass


class ModReacher3DEnv(AnomMJEnv, Reacher3DEnv):
    def __init__(self, mod, anomaly_delay):
        self.ep_length = 150
        kwargs = {}
        if mod == 1 or mod == "bm_factor_minor":
            kwargs["bm_factor"] = 1.01
        elif mod == 2 or mod == "bm_factor_severe":
            kwargs["bm_factor"] = 1.5
        elif mod == 3 or mod == "act_factor_minor":
            kwargs["act_factor"] = 1.01
        elif mod == 4 or mod == "act_factor_severe":
            kwargs["act_factor"] = 1.5
        elif mod == 5 or mod == "act_offset_minor":
            kwargs["act_offset"] = 0.01
        elif mod == 6 or mod == "act_offset_severe":
            kwargs["act_offset"] = -0.5
        elif mod == 7 or mod == "act_noise_minor":
            kwargs["act_noise"] = 0.01
        elif mod == 8 or mod == "act_noise_severe":
            kwargs["act_noise"] = 1
        elif mod == 9 or mod == "force_vector_minor":
            kwargs["force_vec_dict"] = {"r_forearm_link": [0, 0, -0.5, 0, 0, 0]}
        elif mod == 10 or mod == "force_vector_severe":
            kwargs["force_vec_dict"] = {"r_forearm_link": [0, 0, -5, 0, 0, 0]}
        else:
            raise ValueError("mod not speciified")
        if anomaly_delay == "random":
            anomaly_delay = np.random.randint(2, self.ep_length)
        super().__init__(injection_time=anomaly_delay, **kwargs)
