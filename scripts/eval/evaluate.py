# Evaluation script for all methods including DeepEnsemble.

import torch
import numpy as np
import json
import argparse
import os
from collections import defaultdict

from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import VanillaMLP, RecurrentPolicy, LargeMLPPolicy
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, EnvConfig

CHECKPOINT_DIR = "checkpoints"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="ensemble",
                        choices=["ensemble", "vanilla", "lstm", "gru", "large_mlp"])