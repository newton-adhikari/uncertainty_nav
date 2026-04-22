# Evaluation script for all methods including DeepEnsemble.

import torch
import numpy as np
import json
import argparse
import os
from collections import defaultdict

from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import VanillaMLP, RecurrentPolicy, LargeMLPPolicy
from uncertainty_nav.mc_dropout import MCDropoutPolicy
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D, EnvConfig

CHECKPOINT_DIR = "checkpoints"


def load_policy(policy_type, obs_dim, act_dim, checkpoint, device, n_members=5, mc_samples=20):
    if policy_type == "ensemble":
        paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(n_members)]
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            print(f"[WARNING] No ensemble member checkpoints found in {CHECKPOINT_DIR}/")
            return None
        policy = DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)
        print(f"  Loaded ensemble with {len(existing)} members")
        return policy
    elif policy_type == "mc_dropout":
        policy = MCDropoutPolicy(obs_dim, act_dim).to(device)
        policy._mc_samples = mc_samples  # store T for filename encoding
    elif policy_type in ("lstm", "gru"):
        policy = RecurrentPolicy(obs_dim, act_dim, rnn_type=policy_type).to(device)
    elif policy_type == "large_mlp":
        policy = LargeMLPPolicy(obs_dim, act_dim).to(device)
    else:
        policy = VanillaMLP(obs_dim, act_dim).to(device)

    ckpt = checkpoint or f"{CHECKPOINT_DIR}/{policy_type}_policy.pt"
    if os.path.exists(ckpt):
        policy.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Loaded {policy_type} from {ckpt}")
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt}")
    policy.eval()
    return policy


def run_episode(env, policy, policy_type, device, uncertainty_threshold=0.5):
    obs, _ = env.reset()
    hidden = policy.init_hidden() if policy_type in ("lstm", "gru") else None
    done = False
    total_reward = 0.0
    uncertainties = []
    cautious_steps = 0
    steps = 0

    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if policy_type == "ensemble":
                action, unc, is_cautious = policy.uncertainty_driven_action(
                    obs_t, uncertainty_threshold=uncertainty_threshold
                )
                uncertainties.append(unc.item())
                cautious_steps += int(is_cautious)
            elif policy_type == "mc_dropout":
                mc_T = getattr(policy, '_mc_samples', 20)
                action, unc, is_cautious = policy.uncertainty_driven_action(
                    obs_t, uncertainty_threshold=uncertainty_threshold,
                    n_samples=mc_T
                )
                uncertainties.append(unc.item())
                cautious_steps += int(is_cautious)
            elif policy_type in ("lstm", "gru"):
                action, _, hidden = policy.sample(obs_t, hidden)
            else:
                action, _ = policy.sample(obs_t)

        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    success = info.get("success", False)
    collision = info.get("collision", False)
    timeout = (not success) and (not collision)
    spl = env.compute_spl() if success else 0.0
    return {
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "total_reward": total_reward,
        "spl": spl,
        "steps": steps,
        "path_length": info.get("path_length", 0.0),
        "optimal_path_length": info.get("optimal_path_length", 0.0),
        "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "std_uncertainty": float(np.std(uncertainties)) if uncertainties else 0.0,
        "cautious_step_ratio": cautious_steps / max(steps, 1),
    }


def evaluate(policy_type, env_name, checkpoint, n_episodes=200, n_seeds=5,
             noise_levels=None, output_dir="experiments/results", n_members=5,
             mc_samples=20):
    device = torch.device("cpu")
    env_cfg = ENV_MAP.get(env_name, ENV_A)
    obs_dim = PartialObsNavEnv(env_cfg).observation_space.shape[0]
    act_dim = PartialObsNavEnv(env_cfg).action_space.shape[0]
    policy = load_policy(policy_type, obs_dim, act_dim, checkpoint, device,
                         n_members, mc_samples)
    if policy is None:
        return None

    seed_metrics = defaultdict(list)
    all_episodes = []
    for seed in range(n_seeds):
        env = PartialObsNavEnv(env_cfg, seed=seed + 100)
        episodes = [run_episode(env, policy, policy_type, device)
                    for _ in range(n_episodes // n_seeds)]
        all_episodes.extend(episodes)
        seed_metrics["success_rate"].append(float(np.mean([e["success"] for e in episodes])))
        seed_metrics["collision_rate"].append(float(np.mean([e["collision"] for e in episodes])))
        seed_metrics["timeout_rate"].append(float(np.mean([e["timeout"] for e in episodes])))
        seed_metrics["mean_spl"].append(float(np.mean([e["spl"] for e in episodes])))
        seed_metrics["mean_reward"].append(float(np.mean([e["total_reward"] for e in episodes])))

    metrics = {
        "policy": policy_type,
        "env": env_name,
        "n_episodes": n_episodes,
        "n_seeds": n_seeds,
        "mc_samples": mc_samples if policy_type == "mc_dropout" else None,
        "success_rate": float(np.mean(seed_metrics["success_rate"])),
        "success_rate_std": float(np.std(seed_metrics["success_rate"])),
        "collision_rate": float(np.mean(seed_metrics["collision_rate"])),
        "collision_rate_std": float(np.std(seed_metrics["collision_rate"])),
        "timeout_rate": float(np.mean(seed_metrics["timeout_rate"])),
        "timeout_rate_std": float(np.std(seed_metrics["timeout_rate"])),
        "mean_spl": float(np.mean(seed_metrics["mean_spl"])),
        "mean_spl_std": float(np.std(seed_metrics["mean_spl"])),
        "mean_reward": float(np.mean(seed_metrics["mean_reward"])),
        "mean_path_length": float(np.mean([e["path_length"] for e in all_episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in all_episodes])),
        "mean_uncertainty": float(np.mean([e["mean_uncertainty"] for e in all_episodes])),
        "std_uncertainty": float(np.mean([e["std_uncertainty"] for e in all_episodes])),
        "mean_cautious_ratio": float(np.mean([e["cautious_step_ratio"] for e in all_episodes])),
    }

    # Uncertainty-specific metrics (ensemble and mc_dropout)
    if policy_type in ("ensemble", "mc_dropout"):
        uncs = np.array([e["mean_uncertainty"] for e in all_episodes])
        successes = np.array([e["success"] for e in all_episodes], dtype=float)
        failures = 1.0 - successes

        # Correlation
        corr = float(np.corrcoef(uncs, successes)[0, 1]) if uncs.std() > 1e-8 else 0.0
        metrics["uncertainty_success_correlation"] = corr

        # Quartile calibration
        quartiles = np.percentile(uncs, [25, 50, 75])
        bins = np.digitize(uncs, quartiles)
        calibration = {}
        for b in range(4):
            mask = bins == b
            if mask.sum() > 0:
                calibration[f"q{b}_sr"] = float(successes[mask].mean())
                calibration[f"q{b}_cr"] = float(np.mean([e["collision"] for e, m in zip(all_episodes, mask) if m]))
                calibration[f"q{b}_timeout"] = float(np.mean([e["timeout"] for e, m in zip(all_episodes, mask) if m]))
                calibration[f"q{b}_mean_unc"] = float(uncs[mask].mean())
                calibration[f"q{b}_n"] = int(mask.sum())
        metrics["uncertainty_calibration"] = calibration

        # Per-episode arrays for bootstrap CI computation
        metrics["per_episode_uncertainty"] = [float(u) for u in uncs]
        metrics["per_episode_success"] = [int(s) for s in successes]

        # AUROC
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(failures)) >= 2:
                metrics["auroc_failure"] = float(roc_auc_score(failures, uncs))
            else:
                metrics["auroc_failure"] = 0.5
        except ImportError:
            # Manual AUROC via Mann-Whitney
            pos = uncs[failures == 1]
            neg = uncs[failures == 0]
            if len(pos) > 0 and len(neg) > 0:
                metrics["auroc_failure"] = float(np.mean([float(p > n) + 0.5 * float(p == n)
                                                          for p in pos for n in neg]))
            else:
                metrics["auroc_failure"] = 0.5

        # ECE
        bin_edges = np.linspace(0, uncs.max() + 1e-8, 11)
        ece = 0.0
        for i in range(10):
            mask = (uncs >= bin_edges[i]) & (uncs < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_fail = failures[mask].mean()
            bin_unc = uncs[mask].mean() / (uncs.max() + 1e-8)
            ece += (mask.sum() / len(uncs)) * abs(bin_fail - bin_unc)
        metrics["ece"] = float(ece)

        # Routing table
        routing = []
        for label, max_q in [("No routing (all autonomous)", 3),
                              ("Route Q3 to human", 2),
                              ("Route Q2+Q3 to human", 1),
                              ("Only Q0 autonomous", 0)]:
            mask = bins <= max_q
            routing.append({
                "strategy": label,
                "autonomous_frac": float(mask.mean()),
                "autonomous_sr": float(successes[mask].mean()) if mask.sum() > 0 else 0.0,
                "human_burden": float((~mask).mean()),
            })
        metrics["routing_table"] = routing

    if noise_levels is not None:
        robustness = {}
        for noise in noise_levels:
            noisy_cfg = EnvConfig(
                map_size=env_cfg.map_size,
                laser_noise_std=noise,
                occlusion_prob=env_cfg.occlusion_prob,
                fov_deg=env_cfg.fov_deg,
                n_static_obstacles=env_cfg.n_static_obstacles,
                n_dynamic_obstacles=env_cfg.n_dynamic_obstacles,
                dynamic_speed=env_cfg.dynamic_speed,
                interior_walls=env_cfg.interior_walls,
                max_steps=env_cfg.max_steps,
            )
            noisy_env = PartialObsNavEnv(noisy_cfg)
            eps = [run_episode(noisy_env, policy, policy_type, device) for _ in range(50)]
            robustness[str(noise)] = float(np.mean([e["success"] for e in eps]))
        metrics["robustness_curve"] = robustness

    os.makedirs(output_dir, exist_ok=True)
    if policy_type == "mc_dropout":
        T = getattr(policy, '_mc_samples', mc_samples)
        out_path = f"{output_dir}/{policy_type}_T{T}_env{env_name}.json"
    else:
        out_path = f"{output_dir}/{policy_type}_env{env_name}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Policy: {policy_type} | Env: {env_name} | Seeds: {n_seeds}")
    print(f"  Success Rate:    {metrics['success_rate']:.3f} ± {metrics['success_rate_std']:.3f}")
    print(f"  SPL:             {metrics['mean_spl']:.3f} ± {metrics['mean_spl_std']:.3f}")
    print(f"  Collision Rate:  {metrics['collision_rate']:.3f} ± {metrics['collision_rate_std']:.3f}")
    print(f"  Mean Reward:     {metrics['mean_reward']:.2f}")
    if policy_type == "ensemble":
        print(f"  Mean Uncertainty:{metrics['mean_uncertainty']:.4f}")
        print(f"  Unc-Success Corr:{metrics.get('uncertainty_success_correlation', 0.0):.3f}")
        cal = metrics.get("uncertainty_calibration", {})
        if cal:
            print(f"  Calibration (low→high unc quartiles):")
            for b in range(4):
                print(f"    Q{b}: unc={cal.get(f'q{b}_mean_unc',0):.3f}  SR={cal.get(f'q{b}_sr',0):.3f}")
    print(f"{'='*50}\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="ensemble",
                        choices=["ensemble", "mc_dropout", "vanilla", "lstm", "gru", "large_mlp"])
    parser.add_argument("--env", default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_members", type=int, default=5)
    parser.add_argument("--mc_samples", type=int, default=20,
                        help="Number of MC-Dropout forward passes (T)")
    parser.add_argument("--noise_sweep", action="store_true")
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30] if args.noise_sweep else None
    evaluate(args.policy, args.env, args.checkpoint, args.n_episodes,
             args.n_seeds, noise_levels, args.output_dir,
             n_members=args.n_members, mc_samples=args.mc_samples)
