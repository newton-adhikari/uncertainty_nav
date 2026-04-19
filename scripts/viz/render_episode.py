# Renders a clean matplotlib animation showing:
# Robot trajectory color-coded by epistemic uncertainty (green→red)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import animation
import argparse
import os
import math


from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import VanillaMLP
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D, EnvConfig

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "experiments/videos"
ENV_MAP = {"A": ENV_A, "B": ENV_B, "C": ENV_C, "D": ENV_D}

# ── Color utilities ──────────────────────────────────────────────────

def uncertainty_color(u, u_max=1.0):
    # Map uncertainty to green→yellow→red
    t = min(u / u_max, 1.0)
    if t < 0.5:
        return (2 * t, 1.0, 0.0, 0.9)       # green → yellow
    else:
        return (1.0, 2 * (1 - t), 0.0, 0.9)  # yellow → red


def load_ensemble(obs_dim, act_dim, device, n_members=5):
    paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(n_members)]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        print("[ERROR] No ensemble checkpoints found")
        return None
    return DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)


def load_vanilla(obs_dim, act_dim, device):
    policy = VanillaMLP(obs_dim, act_dim).to(device)
    ckpt = f"{CHECKPOINT_DIR}/vanilla_policy.pt"
    if os.path.exists(ckpt):
        policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()
    return policy

