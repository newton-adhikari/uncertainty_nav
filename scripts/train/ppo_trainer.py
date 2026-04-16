import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import json


class PPOTrainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
