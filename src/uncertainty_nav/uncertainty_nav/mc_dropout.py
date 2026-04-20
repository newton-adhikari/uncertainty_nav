import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class MCDropoutPolicy(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(obs)
        mean = self.mean_head(feat)
        return mean, self.log_std.expand_as(mean)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob

    def mc_forward(self, obs: torch.Tensor, n_samples: int = 20) -> dict:
        # Run K stochastic forward passes with dropout ON
        self.train()  # keep dropout active
        means = []
        for _ in range(n_samples):
            m, _ = self.forward(obs)
            means.append(m)
        self.eval()

        means_stack = torch.stack(means, dim=0)  # (K, batch, action_dim)
        mc_mean = means_stack.mean(0)
        mc_var = means_stack.var(0)
        epistemic_uncertainty = mc_var.mean(-1)  # scalar per sample

        return {
            "action": mc_mean,
            "epistemic_uncertainty": epistemic_uncertainty,
            "member_means": means_stack,
        }
    
    
