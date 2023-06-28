# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
from ood_baselines.common.base_detector import Base_Detector
from scipy.special import ndtr
from scipy.stats import multivariate_normal


class GAUSSIAN_Detector(Base_Detector):
    def __init__(self, env, criterion="pdf", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion

    def _predict_scores(self, obs, *args, **kwargs) -> np.ndarray:
        if self.criterion == "pdf":
            # Log of the probability density function evaluated at `x`
            log_p = self.model.logpdf(obs)
            return -log_p
        elif self.criterion == "cdf":
            z = (obs - self.mu) / self.std
            # p = multivariate_normal.cdf(
            #     -np.abs(z),
            #     mean=np.zeros(obs.shape[-1]),
            #     cov=np.eye(obs.shape[-1]),
            # )
            # this is the same as mvn.cdf()
            p = np.prod(ndtr(-np.abs(z)), axis=1)
            return -p

    def _fit(self, train_ep_data, val_ep_data, *args, **kwargs) -> None:
        X_train = np.concatenate([ep.states for ep in train_ep_data], axis=0)
        self.mu = np.mean(X_train, axis=0)
        self.cov = np.cov(X_train, rowvar=False)
        self.std = np.std(X_train, axis=0, keepdims=True)
        self.model = multivariate_normal(cov=self.cov, mean=self.mu, allow_singular=True)  #
