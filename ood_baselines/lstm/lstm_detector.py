# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os

import numpy as np
import torch
from ood_baselines.common.base_detector import Base_Detector
from ood_baselines.lstm.lstm_dm import LSTM_DM


class LSTM_Detector(Base_Detector):
    def __init__(self, env, use_acts=False, model_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if use_acts:
            input_size = env.observation_space.shape[0] + env.action_space.shape[0]
        else:
            input_size = env.observation_space.shape[0]
        output_size = env.observation_space.shape[0]
        self.model = LSTM_DM(input_size=input_size, output_size=output_size, use_acts=use_acts, **model_kwargs)

    def _predict_scores(self, obs, acts):
        if obs.ndim == 2:
            obs = np.expand_dims(obs, 0)
        if acts.ndim == 2:
            acts = np.expand_dims(acts, 0)

        preds = self.model.predict_next_states(obs[:, :-1, :], acts)
        targets = obs[:, 1:, :]

        return self._mse(preds, targets)

    def _mse(self, preds, targets):
        sq_error = (preds - targets) ** 2
        mse = np.mean(sq_error.squeeze(), axis=-1)
        return mse

    def _fit(self, train_ep_data, val_ep_data, n_train_epochs=1000, *args, **kwargs):
        self.model.fit(
            train_ep_data=train_ep_data,
            val_ep_data=val_ep_data,
            n_train_epochs=n_train_epochs,
            *args,
            **kwargs
        )
