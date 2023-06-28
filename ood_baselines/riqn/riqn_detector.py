# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import os
from locale import normalize

import numpy as np
import torch
from ood_baselines.common.base_detector import Base_Detector
from .riqn.riqn_predictor import RIQN_Predictor


class RIQN_Detector(Base_Detector):
    def __init__(self, env, model=None, horizon=1, device=None, normalize_data=True, model_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.device = torch.device("cuda") or device
        self.normalize_data = normalize_data
        self.model = RIQN_Predictor(input_features=env.observation_space.shape[-1], **model_kwargs)

    def _predict_scores(self, obs, acts):
        return self.model.predict_episode(obs)

    def _fit(self, train_ep_data, val_ep_data, n_train_epochs=1_000, *args, **kwargs):
        X_train = np.array([ep.states for ep in train_ep_data])
        self.model.fit(train_ep_obs=X_train, epochs=n_train_epochs, *args, **kwargs)
