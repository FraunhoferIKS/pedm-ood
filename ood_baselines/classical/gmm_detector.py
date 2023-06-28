# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
from ood_baselines.common.base_detector import Base_Detector
from sklearn.mixture import GaussianMixture


class GMM_Detector(Base_Detector):
    def __init__(self, env, n_components=3, n_init=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_components = n_components
        self.n_init = n_init

    def _predict_scores(self, obs, *args, **kwargs) -> np.ndarray:
        scores = self.model.score_samples(obs)
        return -scores

    def _fit(self, train_ep_data, val_ep_data, *args, **kwargs) -> None:
        X_train = np.concatenate([ep.states for ep in train_ep_data], axis=0)
        self.model = GaussianMixture(n_components=self.n_components, n_init=self.n_init)
        self.model.fit(X_train)
