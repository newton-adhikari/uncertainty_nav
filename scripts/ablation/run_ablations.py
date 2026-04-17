# Ablation studies using DeepEnsemble (independently trained members).
# TODO:  create 5 ablation for different policies

import torch
import numpy as np
import json
import os
from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import LargeMLPPolicy
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_B

CHECKPOINT_DIR = "checkpoints"


def _load_ensemble(n_members, device):
    """Load a DeepEnsemble with n_members from independently trained checkpoints."""
    paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(n_members)]
    existing = [p for p in paths if os.path.exists(p)]
    if len(existing) < n_members:
        print(f"[WARNING] Only {len(existing)}/{n_members} member checkpoints found")
    if not existing:
        return None
    env = PartialObsNavEnv(ENV_B)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)


def _load_baseline(policy_class, checkpoint, obs_dim, act_dim, device):
    policy = policy_class(obs_dim, act_dim).to(device)
    if os.path.exists(checkpoint):
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint}")
        return None
    policy.eval()
    return policy


def run_episodes(env_cfg, policy, policy_type, device, n=500, threshold=0.5,
                 use_uncertainty_action=True, n_seeds=5):
    seed_results = []
    eps_per_seed = n // n_seeds
    for seed in range(n_seeds):
        env = PartialObsNavEnv(env_cfg, seed=seed * 1000 + 100)
        episodes = []
        for _ in range(eps_per_seed):
            obs, _ = env.reset()
            done, uncs, cautious = False, [], 0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    if policy_type == "ensemble":
                        if use_uncertainty_action:
                            action, unc, is_c = policy.uncertainty_driven_action(
                                obs_t, uncertainty_threshold=threshold)
                            uncs.append(unc.item())
                            cautious += int(is_c)
                        else:
                            out = policy(obs_t)
                            action = out["action"]
                            uncs.append(out["epistemic_uncertainty"].item())
                    else:
                        action, _ = policy.sample(obs_t)
                obs, _, term, trunc, info = env.step(action.squeeze(0).cpu().numpy())
                done = term or trunc
            spl = env.compute_spl() if info.get("success") else 0.0
            episodes.append({
                "success": info.get("success", False),
                "collision": info.get("collision", False),
                "spl": spl,
                "mean_uncertainty": float(np.mean(uncs)) if uncs else 0.0,
                "cautious_ratio": cautious / max(env._step, 1),
            })
        seed_results.append(episodes)
    return seed_results


def bootstrap_ci(values, n_bootstrap=2000, ci=0.95):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boot = np.sort([np.mean(rng.choice(values, len(values), replace=True))
                    for _ in range(n_bootstrap)])
    return float(boot[int((1-ci)/2*n_bootstrap)]), float(boot[int((1+ci)/2*n_bootstrap)])


def compute_stats(seed_results):
    all_eps = [e for s in seed_results for e in s]
    per_seed_sr = [float(np.mean([e["success"] for e in s])) for s in seed_results]
    successes = [float(e["success"]) for e in all_eps]
    lo, hi = bootstrap_ci(successes)
    return {
        "success_rate": float(np.mean(per_seed_sr)),
        "success_rate_std": float(np.std(per_seed_sr)),
        "success_rate_ci95": [lo, hi],
        "collision_rate": float(np.mean([e["collision"] for e in all_eps])),
        "mean_spl": float(np.mean([e["spl"] for e in all_eps])),
        "mean_uncertainty": float(np.mean([e["mean_uncertainty"] for e in all_eps])),
        "cautious_ratio": float(np.mean([e["cautious_ratio"] for e in all_eps])),
        "n_episodes": len(all_eps),
    }


def mann_whitney_test(results_a, results_b):
    from scipy import stats as sp
    a = [float(e["success"]) for s in results_a for e in s]
    b = [float(e["success"]) for s in results_b for e in s]
    _, p = sp.mannwhitneyu(a, b, alternative='two-sided')
    return float(p)


def _save(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def ablation_size_vs_uncertainty(device, output_dir="experiments/results"):
    pass


def ablation_uncertainty_action_vs_mean(device, output_dir="experiments/results"):
    pass


def ablation_ensemble_size(device, output_dir="experiments/results"):
    pass


def ablation_threshold_sensitivity(device, output_dir="experiments/results"):
    pass


if __name__ == "__main__":
    device = torch.device("cpu")
    print("=" * 60)
    print(" Running ablation studies ")
    print("=" * 60)
    ablation_size_vs_uncertainty(device)
    ablation_uncertainty_action_vs_mean(device)
    ablation_ensemble_size(device)
    ablation_threshold_sensitivity(device)
    print("\nAll ablations complete.")