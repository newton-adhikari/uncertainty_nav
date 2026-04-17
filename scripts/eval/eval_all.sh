#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo " Evaluating all methods"
echo "=========================================="

# Ensemble (assembled from independently trained members)
for env in A B; do
    echo ""
    echo "--- ensemble on Env $env ---"
    python3 scripts/eval/evaluate.py --policy ensemble --env "$env" \
        --n_episodes 200 --n_seeds 5 --n_members 5 --noise_sweep
done

# Baselines
for method in vanilla lstm gru large_mlp; do
    for env in A B; do
        checkpoint="checkpoints/${method}_policy.pt"
        echo ""
        echo "--- $method on Env $env ---"
        python3 scripts/eval/evaluate.py --policy "$method" --env "$env" \
            --checkpoint "$checkpoint" --n_episodes 200 --n_seeds 5 --noise_sweep
    done
done

echo ""
echo "Generating plots..."
python3 scripts/eval/plot_results.py

echo ""
echo "Done. Results in experiments/results/, plots in experiments/plots/"