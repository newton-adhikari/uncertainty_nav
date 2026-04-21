# Compute pairwise cosine similarity between ensemble member predictions

import torch
import numpy as np
import json
import argparse
import os

from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D

CHECKPOINT_DIR = "checkpoints"
ENV_MAP = {"A": ENV_A, "B": ENV_B, "C": ENV_C, "D": ENV_D}


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)


def collect_member_predictions(env, ensemble, device, n_episodes=200, n_seeds=5):
    # Run episodes and collect per-step predictions from each ensemble member.
    episode_stats = []

    for seed in range(n_seeds):
        env_instance = PartialObsNavEnv(env, seed=seed + 300)
        eps_per_seed = n_episodes // n_seeds

        for _ in range(eps_per_seed):
            obs, _ = env_instance.reset()
            done = False
            step_cosines = []
            step_uncertainties = []

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = ensemble.forward(obs_t)
                    member_means = out["member_means"]  # (N, 1, action_dim)
                    uncertainty = out["epistemic_uncertainty"].item()

                # Compute pairwise cosine similarity between members
                members_np = member_means.squeeze(1).cpu().numpy()  # (N, action_dim)
                n_members = members_np.shape[0]
                pairwise_cos = []
                for i in range(n_members):
                    for j in range(i + 1, n_members):
                        pairwise_cos.append(cosine_similarity(members_np[i], members_np[j]))

                step_cosines.append(float(np.mean(pairwise_cos)))
                step_uncertainties.append(uncertainty)

                # Take action (use ensemble mean)
                action = out["action"].squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = env_instance.step(action)
                done = terminated or truncated

            success = info.get("success", False)
            episode_stats.append({
                "mean_cosine_similarity": float(np.mean(step_cosines)),
                "std_cosine_similarity": float(np.std(step_cosines)),
                "mean_uncertainty": float(np.mean(step_uncertainties)),
                "success": success,
            })

    return episode_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_members", type=int, default=5)
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()

    device = torch.device("cpu")
    obs_dim = 39
    act_dim = 2

    # Load ensemble
    paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(args.n_members)]
    existing = [p for p in paths if os.path.exists(p)]
    ensemble = DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)
    print(f"Loaded ensemble with {len(existing)} members")

    results = {}
    for env_name in args.envs:
        print(f"\n{'='*50}")
        print(f"Environment {env_name}")
        print(f"{'='*50}")

        env_cfg = ENV_MAP[env_name]
        stats = collect_member_predictions(env_cfg, ensemble, device,
                                           args.n_episodes, args.n_seeds)

        all_cosines = [s["mean_cosine_similarity"] for s in stats]
        success_cosines = [s["mean_cosine_similarity"] for s in stats if s["success"]]
        failure_cosines = [s["mean_cosine_similarity"] for s in stats if not s["success"]]

        results[env_name] = {
            "mean_cosine_similarity": float(np.mean(all_cosines)),
            "std_cosine_similarity": float(np.std(all_cosines)),
            "success_mean_cosine": float(np.mean(success_cosines)) if success_cosines else None,
            "failure_mean_cosine": float(np.mean(failure_cosines)) if failure_cosines else None,
            "n_episodes": len(stats),
            "n_success": len(success_cosines),
            "n_failure": len(failure_cosines),
            "success_rate": float(np.mean([s["success"] for s in stats])),
        }

        print(f"  Overall cosine sim:  {results[env_name]['mean_cosine_similarity']:.4f} "
              f"± {results[env_name]['std_cosine_similarity']:.4f}")
        if success_cosines:
            print(f"  Success episodes:    {results[env_name]['success_mean_cosine']:.4f}")
        if failure_cosines:
            print(f"  Failure episodes:    {results[env_name]['failure_mean_cosine']:.4f}")
        print(f"  SR: {results[env_name]['success_rate']:.3f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/ensemble_cosine_similarity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
