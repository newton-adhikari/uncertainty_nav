# Baseline models for comparison
# VanillaPPO
# RecurrentPolicy
# LargeMLPPolicy


import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class VanillaMLP(nn.Module):
    # Standard MLP policy — PPO or SAC actor. No memory, no uncertainty.

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(obs)
        mean = self.mean_head(feat)
        return mean, self.log_std.expand_as(mean)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)

        # Tanh-corrected log_prob
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob


class RecurrentPolicy(nn.Module):
    # LSTM/GRU policy — handles partial observability via hidden state.


    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 256,
        rnn_type: str = "lstm",  # "lstm" or "gru"  as other options
    ):
        super().__init__()
        self.hidden_size = hidden
        self.rnn_type = rnn_type

        self.encoder = nn.Linear(obs_dim, hidden)
        if rnn_type == "lstm":
            self.rnn = nn.LSTMCell(hidden, hidden)
        else:
            self.rnn = nn.GRUCell(hidden, hidden)

        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # std≈0.6, this is for less noisy start

    def init_hidden(self, batch_size: int = 1) -> Tuple:
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        if self.rnn_type == "lstm":
            return (h, torch.zeros_like(h))
        return (h,)

    def forward(
        self, obs: torch.Tensor, hidden: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        feat = torch.relu(self.encoder(obs))
        if self.rnn_type == "lstm":
            h, c = self.rnn(feat, hidden)
            new_hidden = (h, c)
        else:
            h = self.rnn(feat, hidden[0])
            new_hidden = (h,)
        mean = self.mean_head(h)
        return mean, self.log_std.expand_as(mean), new_hidden

    def sample(
        self, obs: torch.Tensor, hidden: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        mean, log_std, new_hidden = self.forward(obs, hidden)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)

        # Tanh-corrected log_prob 
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob, new_hidden


class LargeMLPPolicy(nn.Module):
    
    # Ablation baseline: MLP with same parameter count as EpistemicEnsemble.
    # here it answers, is improvement from uncertainty or just model size?"
    

    def __init__(self, obs_dim: int, action_dim: int, n_members: int = 5, hidden: int = 256):
        # Match ensemble param count: n_members * (obs->hidden->hidden->action)
        large_hidden = int(hidden * (n_members ** 0.5))
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, large_hidden), nn.ReLU(),
            nn.Linear(large_hidden, large_hidden), nn.ReLU(),
            nn.Linear(large_hidden, large_hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(large_hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(obs)
        mean = self.mean_head(feat)
        return mean, self.log_std.expand_as(mean)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        
        # Tanh-corrected log_prob
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob
