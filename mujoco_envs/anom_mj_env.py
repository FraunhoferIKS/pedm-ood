# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np


class AnomMJEnv:
    def __init__(
        self,
        bm_factor=1,
        force_vec_dict={},
        act_factor=None,
        act_offset=None,
        act_noise=None,
        obs_factor=None,
        obs_offset=None,
        obs_noise=None,
        injection_time=100,
    ):
        self.step_counter = 0
        self.injection_time = injection_time
        super().__init__()
        self.nominal_bm = self.model.body_mass.copy()
        self.bm_factor = bm_factor
        self.force_vec_dict = force_vec_dict
        self.act_factor = act_factor
        self.act_offset = act_offset
        self.act_noise = act_noise
        self.obs_factor = obs_factor
        self.obs_offset = obs_offset
        self.obs_noise = obs_noise

    def dist_parameters(self):
        self.model.body_mass[:] = self.nominal_bm[:] * self.bm_factor
        for body_name, f_vec in self.force_vec_dict.items():
            body_id = self.sim.model.body_name2id(body_name)
            self.sim.data.xfrc_applied[body_id][:] = f_vec[:]

    def reset_parameters(self):
        self.model.body_mass[:] = self.nominal_bm[:]
        self.force_applied = np.zeros_like(self.sim.data.xfrc_applied)

    def step(self, act):

        if self.step_counter < self.injection_time:
            self.step_counter += 1
            return super().step(act)

        else:
            if self.step_counter == self.injection_time:
                self.dist_parameters()

            if self.act_offset is not None:
                act = act + self.act_offset
            if self.act_factor is not None:
                act = act * self.act_factor
            if self.act_noise is not None:
                act = np.random.normal(act, self.act_noise)

            obs, reward, done, info = super().step(act)

            if self.obs_offset is not None:
                obs = obs + self.obs_offset
            if self.obs_factor is not None:
                obs = obs * self.obs_factor
            if self.obs_noise is not None:
                obs = np.random.normal(obs, self.obs_noise)

            self.step_counter += 1
            return obs, reward, done, info

    def reset(self):
        self.reset_parameters()
        self.step_counter = 0
        return super().reset()
