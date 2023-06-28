# ----------------------------------------------------------------------------------------------------------------------
# Copyright©[2023] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its
# Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
# This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html).
#
# Contact: tom.haider@iks.fraunhofer.de
# ----------------------------------------------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM_Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, n_lstm_layers=8, fc_size=128, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_lstm_layers, batch_first=True, dropout=dropout
        )
        self.fc_1 = nn.Linear(hidden_size, fc_size)  # fully connected 1
        self.fc_out = nn.Linear(fc_size, output_size)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x, h_0=None, c_0=None):
        """
        Args:
            x: ndarray[batch_size, sequence_length, state_dim]
            h_o: ndarray[n_lstm_layers, batch_size, hidden_size]
            c_0: ndarray[n_lstm_layers, batch_size, hidden_size]
        Returns:
            out: ndarray[batch_size, sequence_length, state_dim]
            h_o: ndarray[n_lstm_layers, batch_size, hidden_size]
            c_0: ndarray[n_lstm_layers, batch_size, hidden_size]
        """

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device).float()

        if h_0 is None and c_0 is None:
            h_0, c_0 = self._init_hidden_state(x.size(0))

        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and cell state
        out = self.relu(lstm_out)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_out(out)  # final out
        return out, (hn, cn)

    def _init_hidden_state(self, batch_size):
        h_0 = Variable(torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size)).to(self.device)  # cell state
        return h_0, c_0


class LSTM_Model:
    def __init__(self, lr=0.005, *args, **kwargs):
        self.net = LSTM_Net(*args, **kwargs)
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.net.device)

    def fit(self, train_set, val_set, n_train_epochs, batch_size=16, val_every=25, verbose=True):
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)
        for epo in range(n_train_epochs):
            self.train_epoch(train_loader)
            if not epo % val_every:
                val_loss = self.val_epoch(val_loader)
                if verbose:
                    print(epo, val_loss)

    def train_epoch(self, train_loader):
        self.net.train()
        for i, (state, next_state) in enumerate(train_loader):
            state, next_state = state.to(self.net.device).float(), next_state.to(self.net.device).float()
            output, _ = self.net(state)
            loss = self.criterion(output, next_state)
            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def val_epoch(self, val_loader):
        self.net.eval()
        val_losses = []
        for state, next_state in val_loader:
            state, next_state = state.to(self.net.device).float(), next_state.to(self.net.device).float()
            output, _ = self.net(state)
            loss = self.criterion(output, next_state)
            val_losses.append(loss.item())
        self.net.train()
        return np.mean(val_losses)

    # @torch.no_grad()
    # def predict_n_states(self, states, h_0=None, c_0=None):
    #     """
    #     Args:
    #         states: ndarray[batch_size, sequence_length, state_dim]
    #     Returns:
    #         next_states: ndarray[batch_size, sequence_length, state_dim]
    #     """
    #     next_states, _ = self.net.forward(states, h_0=h_0, c_0=c_0)
    #     return next_states.cpu().detach().numpy()

    def save(self, path):
        print("saving LSTM model at: ", path)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "init_kwargs": {
                    "input_size": self.net.input_size,
                    "output_size": self.net.output_size,
                    "hidden_size": self.net.hidden_size,
                    "n_lstm_layers": self.net.n_lstm_layers,
                    "dropout": self.net.dropout,
                },
                "save_attrs": {},
            },
            f=path,
        )

    @classmethod
    def load(cls, path):
        print("loading LSTM model from: ", path)
        saved_variables = torch.load(path)
        model = cls(**saved_variables["init_kwargs"])
        model.net.load_state_dict(saved_variables["state_dict"])
        model._save_attrs = saved_variables["save_attrs"]
        for k, v in model._save_attrs.items():
            setattr(model, k, v)
        return model
