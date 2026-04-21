# temperature scaling for uncertainty calibration.

import torch
import numpy as np
import json
import argparse
import os
from scipy.optimize import minimize_scalar

from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.mc_dropout import MCDropoutPolicy
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D

CHECKPOINT_DIR = "checkpoints"
ENV_MAP = {"A": ENV_A, "B": ENV_B, "C": ENV_C, "D": ENV_D}


def run_episode_collect(env, policy, policy_type, device, mc_samples=20):
    # Run one episode, return (mean_uncertainty, success)
    obs, _ = env.reset()
    done = False
    uncertainties = []

    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if policy_type == "ensemble":
                action, unc, _ = policy.uncertainty_driven_action(obs_t)
                uncertainties.append(unc.item())
            elif policy_type == "mc_dropout":
                action, unc, _ = policy.uncertainty_driven_action(
                    obs_t, n_samples=mc_samples)
                uncertainties.append(unc.item())

        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

    return float(np.mean(uncertainties)), info.get("success", False)


def compute_ece(uncertainties, failures, n_bins=10):
    # Compute Expected Calibration Error
    uncs = np.array(uncertainties)
    fails = np.array(failures, dtype=float)
    if uncs.max() < 1e-10:
        return 1.0
    bin_edges = np.linspace(0, uncs.max() + 1e-8, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (uncs >= bin_edges[i]) & (uncs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_fail = fails[mask].mean()
        bin_unc_norm = uncs[mask].mean() / (uncs.max() + 1e-8)
        ece += (mask.sum() / len(uncs)) * abs(bin_fail - bin_unc_norm)
    return float(ece)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def calibrated_ece(uncertainties, failures, temperature, n_bins=10):
    # Compute ECE after temperature scaling.
    
    uncs = np.array(uncertainties)
    fails = np.array(failures, dtype=float)
    u_med = np.median(uncs) + 1e-10
    # Calibrated probability of failure
    logits = np.log(uncs / u_med + 1e-10) / (temperature + 1e-10)
    p_fail = sigmoid(logits)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p_fail >= bin_edges[i]) & (p_fail < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_actual_fail = fails[mask].mean()
        bin_predicted_fail = p_fail[mask].mean()
        ece += (mask.sum() / len(uncs)) * abs(bin_actual_fail - bin_predicted_fail)
    return float(ece)


def find_optimal_temperature(cal_uncs, cal_failures):
    # Find temperature that minimizes ECE on calibration set
    def objective(T):
        return calibrated_ece(cal_uncs, cal_failures, T)

    result = minimize_scalar(objective, bounds=(0.01, 10.0), method='bounded')
    return result.x, result.fun


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="mc_dropout", choices=["mc_dropout", "ensemble"])
    parser.add_argument("--envs", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--mc_samples", type=int, default=20)
    parser.add_argument("--cal_fraction", type=float, default=0.5,
                        help="Fraction of episodes used for calibration (rest for test)")
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()

    device = torch.device("cpu")
    obs_dim = 39
    act_dim = 2

    # Load policy
    if args.policy == "mc_dropout":
        policy = MCDropoutPolicy(obs_dim, act_dim).to(device)
        ckpt = f"{CHECKPOINT_DIR}/mc_dropout_policy.pt"
        policy.load_state_dict(torch.load(ckpt, map_location=device))
        policy._mc_samples = args.mc_samples
    elif args.policy == "ensemble":
        paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(5)]
        policy = DeepEnsemble.from_checkpoints(paths, obs_dim, act_dim, device=device)

    results = {}
    for env_name in args.envs:
        print(f"\n{'='*50}")
        print(f"Environment {env_name}")
        print(f"{'='*50}")

        env_cfg = ENV_MAP[env_name]

        # Collect episodes
        all_uncs = []
        all_failures = []
        for seed in range(args.n_seeds):
            env = PartialObsNavEnv(env_cfg, seed=seed + 200)  # different seeds from main eval
            eps_per_seed = args.n_episodes // args.n_seeds
            for _ in range(eps_per_seed):
                unc, success = run_episode_collect(
                    env, policy, args.policy, device, args.mc_samples)
                all_uncs.append(unc)
                all_failures.append(0.0 if success else 1.0)

        all_uncs = np.array(all_uncs)
        all_failures = np.array(all_failures)

        # Split into calibration and test
        n_cal = int(len(all_uncs) * args.cal_fraction)
        idx = np.random.RandomState(42).permutation(len(all_uncs))
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        cal_uncs, cal_fails = all_uncs[cal_idx], all_failures[cal_idx]
        test_uncs, test_fails = all_uncs[test_idx], all_failures[test_idx]

        # Raw ECE (on test set)
        raw_ece = compute_ece(test_uncs, test_fails)

        # Find optimal temperature on calibration set
        T_opt, cal_ece = find_optimal_temperature(cal_uncs, cal_fails)

        # Evaluate calibrated ECE on test set
        calibrated_test_ece = calibrated_ece(test_uncs, test_fails, T_opt)

        # AUROC (unchanged by temperature scaling — monotonic transform)
        from sklearn.metrics import roc_auc_score
        if len(np.unique(test_fails)) >= 2:
            auroc = roc_auc_score(test_fails, test_uncs)
        else:
            auroc = 0.5

        results[env_name] = {
            "raw_ece": raw_ece,
            "optimal_temperature": float(T_opt),
            "calibrated_ece": calibrated_test_ece,
            "ece_reduction_pct": float((raw_ece - calibrated_test_ece) / raw_ece * 100),
            "auroc": float(auroc),
            "n_episodes": len(all_uncs),
            "n_cal": n_cal,
            "n_test": len(test_uncs),
            "failure_rate": float(all_failures.mean()),
        }

        print(f"  Failure rate:     {all_failures.mean():.3f}")
        print(f"  Raw ECE:          {raw_ece:.4f}")
        print(f"  Optimal T:        {T_opt:.4f}")
        print(f"  Calibrated ECE:   {calibrated_test_ece:.4f}")
        print(f"  ECE reduction:    {results[env_name]['ece_reduction_pct']:.1f}%")
        print(f"  AUROC:            {auroc:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/temperature_scaling_{args.policy}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
