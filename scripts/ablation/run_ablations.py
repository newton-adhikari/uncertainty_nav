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
    # Ablation 1: Ensemble(N=5) vs LargeMLPPolicy (same param count)
    env = PartialObsNavEnv(ENV_B)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    results, raw = {}, {}
    # Ensemble
    ens = _load_ensemble(5, device)
    if ens:
        sr = run_episodes(ENV_B, ens, "ensemble", device)
        results["ensemble_5"] = compute_stats(sr)
        raw["ensemble_5"] = sr
        print(f"[Ablation 1] ensemble_5: SR={results['ensemble_5']['success_rate']:.3f} "
              f"± {results['ensemble_5']['success_rate_std']:.3f}")

    # Large MLP
    lmlp = _load_baseline(LargeMLPPolicy, f"{CHECKPOINT_DIR}/large_mlp_policy.pt",
                           obs_dim, act_dim, device)
    if lmlp:
        sr = run_episodes(ENV_B, lmlp, "large_mlp", device)
        results["large_mlp"] = compute_stats(sr)
        raw["large_mlp"] = sr
        print(f"[Ablation 1] large_mlp: SR={results['large_mlp']['success_rate']:.3f} "
              f"± {results['large_mlp']['success_rate_std']:.3f}")

    if "ensemble_5" in raw and "large_mlp" in raw:
        try:
            p = mann_whitney_test(raw["ensemble_5"], raw["large_mlp"])
            results["mann_whitney_p_value"] = p
            print(f"[Ablation 1] Mann-Whitney p={p:.4f}")
        except ImportError:
            pass

    _save(results, f"{output_dir}/ablation_size_vs_uncertainty.json")


def ablation_uncertainty_action_vs_mean(device, output_dir="experiments/results"):
    # Ablation 2: Uncertainty-driven action vs mean-only
    ens = _load_ensemble(5, device)
    if not ens:
        return

    results, raw = {}, {}
    for use_unc, name in [(True, "uncertainty_driven"), (False, "mean_only")]:
        sr = run_episodes(ENV_B, ens, "ensemble", device, use_uncertainty_action=use_unc)
        results[name] = compute_stats(sr)
        raw[name] = sr
        print(f"[Ablation 2] {name}: SR={results[name]['success_rate']:.3f} "
              f"± {results[name]['success_rate_std']:.3f}")

    if "uncertainty_driven" in raw and "mean_only" in raw:
        try:
            p = mann_whitney_test(raw["uncertainty_driven"], raw["mean_only"])
            results["mann_whitney_p_value"] = p
            print(f"[Ablation 2] Mann-Whitney p={p:.4f}")
        except ImportError:
            pass

    _save(results, f"{output_dir}/ablation_uncertainty_action.json")


def ablation_ensemble_size(device, output_dir="experiments/results"):
    # Ablation 3: N=1,2,3,5,10 using subsets of independently trained members
    results = {}
    for n in [1, 2, 3, 5, 10]:
        ens = _load_ensemble(n, device)
        if not ens:
            results[f"N={n}"] = {"status": "missing_checkpoints"}
            print(f"[Ablation 3] N={n}: SKIPPED")
            continue
        sr = run_episodes(ENV_B, ens, "ensemble", device)
        stats = compute_stats(sr)
        results[f"N={n}"] = stats
        print(f"[Ablation 3] N={n}: SR={stats['success_rate']:.3f} "
              f"± {stats['success_rate_std']:.3f} "
              f"unc={stats['mean_uncertainty']:.4f}")

    _save(results, f"{output_dir}/ablation_ensemble_size.json")


def ablation_threshold_sensitivity(device, output_dir="experiments/results"):
    # Ablation 4: Threshold sensitivity
    ens = _load_ensemble(5, device)
    if not ens:
        return

    results = {}
    for threshold in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        sr = run_episodes(ENV_B, ens, "ensemble", device, threshold=threshold)
        stats = compute_stats(sr)
        results[f"t={threshold}"] = stats
        print(f"[Ablation 4] threshold={threshold}: SR={stats['success_rate']:.3f} "
              f"cautious={stats['cautious_ratio']:.3f}")

    _save(results, f"{output_dir}/ablation_threshold.json")


if __name__ == "__main__":
    device = torch.device("cpu")
    print("=" * 60)
    print(" Running ablation studies ")
    print("=" * 60)
    ablation_size_vs_uncertainty(device)
    ablation_uncertainty_action_vs_mean(device)

    # NOTE: we have updated :
    # ensemble size ablation is produced separately 
    # evaluate_all_envs.py --auroc_ablation → ensemble_size_auroc.json

    ablation_threshold_sensitivity(device)
    print("\nAll ablations complete.")