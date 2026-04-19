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

def run_episode_trace(env, policy, policy_type, device, threshold=0.5):
    # Run one episode, return full state trace for visualization.
    obs, _ = env.reset()
    done = False
    trace = {
        "robot_x": [], "robot_y": [], "robot_theta": [],
        "goal": (env._goal[0], env._goal[1]),
        "obstacles": [o.copy() for o in env._obstacles],
        "dynamic_obs": [],  # list of lists of (x,y) per step
        "walls": [(w[0].tolist(), w[1].tolist()) for w in env._interior_walls],
        "uncertainty": [],
        "velocity": [],
        "laser_endpoints": [],  # list of (N,2) arrays
        "actions": [],
        "success": False,
        "collision": False,
    }

    while not done:
        # Record state
        x, y, th = env._robot_pose
        trace["robot_x"].append(float(x))
        trace["robot_y"].append(float(y))
        trace["robot_theta"].append(float(th))

        # Record dynamic obstacle positions
        dyn_pos = [(d[0][0], d[0][1]) for d in env._dynamic_obstacles]
        trace["dynamic_obs"].append(dyn_pos)

        # Compute laser endpoints for visualization
        scan_raw = env._get_laser_scans()  # normalized [0,1]
        ranges = scan_raw * env.cfg.max_range
        angles = env._beam_angles_rel + th
        lx = x + ranges * np.cos(angles)
        ly = y + ranges * np.sin(angles)
        trace["laser_endpoints"].append(np.stack([lx, ly], axis=1))

        # Get action
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if policy_type == "ensemble":
                action, unc, is_cautious = policy.uncertainty_driven_action(
                    obs_t, uncertainty_threshold=threshold)
                trace["uncertainty"].append(float(unc.item()))
            else:
                action, _ = policy.sample(obs_t)
                trace["uncertainty"].append(0.0)

        action_np = action.squeeze(0).cpu().numpy()
        v = action_np[0] * env.cfg.max_linear_vel
        trace["velocity"].append(float(abs(v)))
        trace["actions"].append(action_np.copy())

        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

    trace["success"] = info.get("success", False)
    trace["collision"] = info.get("collision", False)
    trace["n_steps"] = len(trace["robot_x"])
    return trace

# to render static snapshot
def render_snapshot():
    # Render a single frame or full trajectory as a figure
    pass
