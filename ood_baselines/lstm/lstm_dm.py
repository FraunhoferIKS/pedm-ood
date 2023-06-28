# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
import torch
from ood_baselines.lstm.lstm_model import LSTM_Model


class LSTM_DM(LSTM_Model):
    """
    LSTM Dynamics Model
    """

    def __init__(self, obs_preproc=None, obs_postproc=None, targ_proc=None, use_acts=True, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.obs_preproc = obs_preproc or (lambda obs: obs)
        self.obs_postproc = obs_postproc or (lambda obs, pred: obs + pred)
        self.targ_proc = targ_proc or (lambda obs, n_obs: n_obs - obs)
        self.use_acts = use_acts

    def fit(self, train_ep_data, val_ep_data, *args, **kwargs):
        X_train, y_train = self._preproc_ep_data(train_ep_data)
        X_val, y_val = self._preproc_ep_data(val_ep_data)
        train_set = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_set = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        return super().fit(train_set, val_set, *args, **kwargs)

    def _preproc_ep_data(self, ep_data):
        stkd_obs = np.array([ep.states for ep in ep_data])[:, :-1, :]
        stkd_n_obs = np.array([ep.states for ep in ep_data])[:, 1:, :]
        stkd_acts = np.array([ep.actions for ep in ep_data])

        if self.use_acts:
            X = np.concatenate([self.obs_preproc(stkd_obs), stkd_acts], axis=-1)
        else:
            X = self.obs_preproc(stkd_obs)

        y = self.targ_proc(stkd_obs, stkd_n_obs)
        return X, y

    @torch.no_grad()
    def predict_next_states(self, obs, acts, h_0=None, c_0=None):
        """
        Args:
            states: ndarray[batch_size, sequence_length, state_dim]
        Returns:
            next_states: ndarray[batch_size, sequence_length, state_dim]
        """

        if self.use_acts:
            inputs = np.concatenate((self.obs_preproc(obs), acts), axis=-1)
        else:
            inputs = self.obs_preproc(obs)

        predictions, _ = self.net.forward(inputs)

        return self.obs_postproc(obs, predictions.cpu().numpy())

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, device="auto"):
        pass
