# we try to implement Epistemic Uncertainty via Deep Ensembles
# by training Train N independent policies from different random seeds


import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PolicyNetwork(nn.Module):
    # Single policy network: maps observation -> (mean, log_std).
    

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Orthogonal init for stable training
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


class DeepEnsemble(nn.Module):
    # Evaluation-only ensemble wrapper. Loads N independently trained
    # PolicyNetwork checkpoints and measures their disagreement.


    def __init__(self, members: list):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.n_members = len(members)

    @staticmethod
    def from_checkpoints(checkpoint_paths: list, obs_dim: int, action_dim: int,
                         hidden: int = 256, device=None):
        # Load N independently trained PolicyNetwork checkpoints.
        members = []
        for path in checkpoint_paths:
            net = PolicyNetwork(obs_dim, action_dim, hidden)
            net.load_state_dict(torch.load(path, map_location=device or "cpu"))
            net.eval()
            members.append(net)
        ensemble = DeepEnsemble(members)
        if device:
            ensemble = ensemble.to(device)
        return ensemble

    def forward(self, obs: torch.Tensor) -> dict:
        means = []
        for member in self.members:
            m, _ = member(obs)
            means.append(m)

        means_stack = torch.stack(means, dim=0)  # (N, batch, action_dim)
        ensemble_mean = means_stack.mean(0)

        if self.n_members > 1:
            epistemic_var = means_stack.var(0)
        else:
            epistemic_var = torch.zeros_like(means_stack[0])

        epistemic_uncertainty = epistemic_var.mean(-1)  # scalar per sample

        return {
            "action": ensemble_mean,
            "epistemic_uncertainty": epistemic_uncertainty,
            "member_means": means_stack,
        }

    def uncertainty_driven_action(
        self, obs: torch.Tensor,
        uncertainty_threshold: float = 0.5,
        caution_scale: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        # Adaptive caution: scale = 1/(1 + alpha*unc), remapped to [caution_scale, 1].
        
        out = self.forward(obs)
        action = out["action"]
        uncertainty = out["epistemic_uncertainty"]

        alpha = 1.0 / (uncertainty_threshold + 1e-8)
        raw_scale = 1.0 / (1.0 + alpha * uncertainty)
        scale = caution_scale + (1.0 - caution_scale) * raw_scale

        is_cautious = scale < 0.9
        action = action * scale.unsqueeze(-1)
        return action, uncertainty, is_cautious.any().item()

    def get_uncertainty_stats(self, obs: torch.Tensor) -> dict:
        with torch.no_grad():
            out = self.forward(obs)
        return {
            "epistemic": out["epistemic_uncertainty"].mean().item(),
        }


class EpistemicEnsemble(DeepEnsemble):
    # Agent-node-compatible ensemble.

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_members: int = 5,
        hidden: int = 256,
    ):
        members = [PolicyNetwork(obs_dim, act_dim, hidden) for _ in range(n_members)]
        super().__init__(members)
        self._obs_dim  = obs_dim
        self._act_dim  = act_dim
        self._hidden   = hidden

    def load_from_dir(self, checkpoint_path: str, device=None) -> int:
        # Returns the number of members successfully loaded.
        
        import os
        dev = device or torch.device("cpu")
        d   = os.path.dirname(os.path.abspath(checkpoint_path))
        loaded = 0
        for i, member in enumerate(self.members):
            p = os.path.join(d, f"ensemble_m{i}_policy.pt")
            if os.path.exists(p):
                state = torch.load(p, map_location=dev, weights_only=True)
                member.load_state_dict(state, strict=True)
                member.eval()
                loaded += 1
        return loaded


class ValueNetwork(nn.Module):
    # Critic for PPO.

    def __init__(self, obs_dim: int, action_dim: int = 0, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1) if action is not None else obs
        return self.net(x)