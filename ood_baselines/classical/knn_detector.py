# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import pickle

import faiss
import numpy as np
from ood_baselines.common.base_detector import Base_Detector


class KNN_Detector(Base_Detector):
    def __init__(self, env, knn_index=None, th=None, k=1, use_acts=False, use_n_obs=False, normalize_data=True):
        self.knn_index = knn_index
        if th:
            self.set_threshold(th)
        self.k = k
        self.use_acts = use_acts
        self.use_n_obs = use_n_obs
        self.normalize_data = normalize_data

    def _predict_scores(self, obs, acts, *args, **kwargs) -> np.ndarray:

        X_test = self.preproc_sample(obs, acts)
        dist, _ = self.knn_index.kneighbors(X_test, k=self.k)
        dist = dist[:, -1].reshape(-1, 1)
        return dist

    def _fit(self, train_ep_data, val_ep_data, *args, **kwargs):

        X_train = self.preproc_ep_data(train_ep_data)
        self.knn_index = FastL2KNN(d=X_train.shape[-1], **kwargs)
        self.knn_index.fit(X_train)

    def preproc_sample(self, obs, acts):
        X_test = obs[:-1]
        if self.use_acts:
            X_test = np.concatenate([X_test, acts], axis=-1)
        if self.use_n_obs:
            n_obs = obs[1:]
            X_test = np.concatenate([X_test, n_obs], axis=-1)
        return X_test

    def preproc_ep_data(self, train_ep_data):
        stkd_obs = np.array([ep.states for ep in train_ep_data])[:, :-1, :]
        X_train = np.concatenate(stkd_obs, axis=0)
        X_train = np.concatenate(stkd_obs, axis=0)
        if self.use_acts:
            acts = np.array([ep.actions for ep in train_ep_data])
            stkd_acts = np.concatenate(acts, axis=0)
            X_train = np.concatenate([X_train, stkd_acts], axis=-1)
        if self.use_n_obs:
            n_obs = np.array([ep.states for ep in train_ep_data])[:, 1:, :]
            stkd_n_obs = np.concatenate(n_obs, axis=0)
            X_train = np.concatenate([X_train, stkd_n_obs], axis=-1)
        return X_train


class FastL2KNN:
    def __init__(self, d, nlist=100, nprobe=1):
        nlist = 100
        quantizer = faiss.IndexFlatL2(d)
        self._index = faiss.IndexIVFFlat(quantizer, d, nlist)
        self._index.nprobe = nprobe

    def fit(self, X):
        self._index.train(X.astype(np.float32))
        self._index.add(X.astype(np.float32))

    def kneighbors(self, X, k):
        k_dist, k_ind = self._index.search(X.astype(np.float32), k=k)
        return k_dist, k_ind

    def save(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, fname):
        with open(fname, "rb") as f:
            model = pickle.load(f)
        return model
