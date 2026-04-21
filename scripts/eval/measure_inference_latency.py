# Measure inference latency for MC-Dropout at various T values.


import torch
import numpy as np
import json
import time
import argparse
import os

from uncertainty_nav.mc_dropout import MCDropoutPolicy
from uncertainty_nav.models import PolicyNetwork, DeepEnsemble

CHECKPOINT_DIR = "checkpoints"


def measure_mc_dropout_latency(policy, obs_dim, device, T_values, n_warmup=50, n_trials=500):
    # Measure latency for MC-Dropout at different T values
    results = {}

    # Generate random observations for benchmarking
    obs_batch = torch.randn(1, obs_dim).to(device)

    for T in T_values:
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                policy.mc_forward(obs_batch, n_samples=T)

        # Timed trials
        latencies = []
        for _ in range(n_trials):
            start = time.perf_counter()
            with torch.no_grad():
                policy.mc_forward(obs_batch, n_samples=T)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        results[f"T={T}"] = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "n_trials": n_trials,
        }
        print(f"  T={T:3d}: mean={results[f'T={T}']['mean_ms']:.3f}ms, "
              f"p95={results[f'T={T}']['p95_ms']:.3f}ms, "
              f"p99={results[f'T={T}']['p99_ms']:.3f}ms")

    return results


def measure_ensemble_latency(ensemble, obs_dim, device, n_warmup=50, n_trials=500):
    # Measure latency for ensemble forward pass
    obs_batch = torch.randn(1, obs_dim).to(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            ensemble.forward(obs_batch)

    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        with torch.no_grad():
            ensemble.forward(obs_batch)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "n_trials": n_trials,
    }


def measure_single_forward_latency(obs_dim, device, n_warmup=50, n_trials=500):
    # Measure latency for a single MLP forward pass (baseline)
    policy = PolicyNetwork(obs_dim, 2).to(device)
    policy.eval()
    obs_batch = torch.randn(1, obs_dim).to(device)

    for _ in range(n_warmup):
        with torch.no_grad():
            policy(obs_batch)

    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        with torch.no_grad():
            policy(obs_batch)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "n_trials": n_trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n_trials", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()

    device = torch.device(args.device)
    obs_dim = 39
    act_dim = 2

    print("="*60)
    print("INFERENCE LATENCY BENCHMARK")
    print(f"Device: {device}")
    print(f"Trials: {args.n_trials}")
    print("="*60)

    # Single forward pass baseline
    print("\n--- Single MLP forward pass ---")
    single_result = measure_single_forward_latency(obs_dim, device, n_trials=args.n_trials)
    print(f"  mean={single_result['mean_ms']:.3f}ms, p95={single_result['p95_ms']:.3f}ms")

    # MC-Dropout at various T
    print("\n--- MC-Dropout ---")
    mc_policy = MCDropoutPolicy(obs_dim, act_dim).to(device)
    ckpt = f"{CHECKPOINT_DIR}/mc_dropout_policy.pt"
    if os.path.exists(ckpt):
        mc_policy.load_state_dict(torch.load(ckpt, map_location=device))
    mc_policy.eval()

    T_values = [1, 5, 10, 15, 20, 30, 50]
    mc_results = measure_mc_dropout_latency(mc_policy, obs_dim, device, T_values,
                                            n_trials=args.n_trials)

    # Ensemble
    print("\n--- Deep Ensemble (N=5) ---")
    paths = [f"{CHECKPOINT_DIR}/ensemble_m{i}_policy.pt" for i in range(5)]
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        ensemble = DeepEnsemble.from_checkpoints(existing, obs_dim, act_dim, device=device)
        ens_result = measure_ensemble_latency(ensemble, obs_dim, device, n_trials=args.n_trials)
        print(f"  mean={ens_result['mean_ms']:.3f}ms, p95={ens_result['p95_ms']:.3f}ms")
    else:
        ens_result = {"error": "No ensemble checkpoints found"}
        print("  [SKIPPED] No ensemble checkpoints found")

    # Summary
    all_results = {
        "device": str(device),
        "obs_dim": obs_dim,
        "n_trials": args.n_trials,
        "single_mlp": single_result,
        "mc_dropout": mc_results,
        "ensemble_N5": ens_result,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/inference_latency_{args.device}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"\nSummary for paper (T ablation table):")
    print(f"  T=5:  {mc_results.get('T=5', {}).get('mean_ms', 'N/A'):.2f} ms")
    print(f"  T=10: {mc_results.get('T=10', {}).get('mean_ms', 'N/A'):.2f} ms")
    print(f"  T=20: {mc_results.get('T=20', {}).get('mean_ms', 'N/A'):.2f} ms")
    print(f"  Ensemble (N=5): {ens_result.get('mean_ms', 'N/A')}")


if __name__ == "__main__":
    main()
