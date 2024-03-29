# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
from ood_baselines.common.base_detector import Base_Detector


class RANDOM_Detector(Base_Detector):
    def __init__(self, *args, **kwargs):
        pass

    def _predict_scores(self, obs, acts, *args, **kwargs) -> np.ndarray:
        return np.random.uniform(0, 1, size=acts.shape[0])

    def _fit(self, *args, **kwargs) -> None:
        pass
