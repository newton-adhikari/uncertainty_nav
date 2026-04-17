# Ablation studies using DeepEnsemble (independently trained members).
# TODO:  create 5 ablation for different policies

import torch
import numpy as np

CHECKPOINT_DIR = "checkpoints"


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