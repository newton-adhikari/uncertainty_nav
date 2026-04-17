# Particle Filter for belief-state estimation (classical baseline).

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Particle:
    x: float
    y: float
    theta: float
    weight: float = 1.0


class ParticleFilter:
    # 2D particle filter for robot localization under partial observability.
    # Motion model: diff drive. Observation model: range sensor likelihood.
    

    def __init__(
        self,
        n_particles: int = 500,
    ):
        self.n = n_particles
 