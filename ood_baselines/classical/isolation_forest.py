# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
from ood_baselines.common.base_detector import Base_Detector
from sklearn.ensemble import IsolationForest


class ISOFOREST_Detector(Base_Detector):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _predict_scores(self, obs, *args, **kwargs) -> np.ndarray:
        df = self.model.decision_function(obs)
        return -df

    def _fit(self, train_ep_data, val_ep_data, *args, **kwargs) -> None:
        X_train = np.concatenate([ep.states for ep in train_ep_data], axis=0)
        self.model = IsolationForest(
            random_state=42, n_jobs=4, max_samples=X_train.shape[0], bootstrap=True, n_estimators=50
        )
        self.model.fit(X_train)
