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
        map_bounds: tuple = (-5.0, 5.0, -5.0, 5.0),  # xmin, xmax, ymin, ymax
        motion_noise: tuple = (0.05, 0.05, 0.02),     # x, y, theta std
        obs_noise_std: float = 0.2,
    ):
        self.n = n_particles
        self.map_bounds = map_bounds
        self.motion_noise = motion_noise
        self.obs_noise_std = obs_noise_std
        self.particles = self._init_uniform()

    def _init_uniform(self) -> list:
        xmin, xmax, ymin, ymax = self.map_bounds
        particles = []
        for _ in range(self.n):
            p = Particle(
                x=np.random.uniform(xmin, xmax),
                y=np.random.uniform(ymin, ymax),
                theta=np.random.uniform(-np.pi, np.pi),
                weight=1.0 / self.n,
            )
            particles.append(p)
        return particles

    def predict(self, v: float, omega: float, dt: float = 0.1):
        # Motion update with noise
        nx, ny, nt = self.motion_noise
        for p in self.particles:
            p.x += (v * np.cos(p.theta) + np.random.normal(0, nx)) * dt
            p.y += (v * np.sin(p.theta) + np.random.normal(0, ny)) * dt
            p.theta += (omega + np.random.normal(0, nt)) * dt

    def update(self, obs_ranges: np.ndarray, expected_ranges_fn):
        # 
        # Observation update: weight particles by likelihood of observed ranges.
        # expected_ranges_fn(x, y, theta) -> np.ndarray of expected ranges
         
        weights = np.zeros(self.n)
        for i, p in enumerate(self.particles):
            expected = expected_ranges_fn(p.x, p.y, p.theta)
            diff = obs_ranges - expected
            log_likelihood = -0.5 * np.sum((diff / self.obs_noise_std) ** 2)
            weights[i] = np.exp(log_likelihood)

        weights += 1e-300  # avoid zero
        weights /= weights.sum()
        for i, p in enumerate(self.particles):
            p.weight = weights[i]

        self._resample()

    def _resample(self):
        # Low-variance resampling
        weights = np.array([p.weight for p in self.particles])
        indices = np.random.choice(self.n, size=self.n, p=weights)
        new_particles = []
        for i in indices:
            p = self.particles[i]
            new_particles.append(Particle(p.x, p.y, p.theta, 1.0 / self.n))
        self.particles = new_particles

    def get_belief_state(self) -> np.ndarray:
        # Returns [mean_x, mean_y, mean_theta, var_x, var_y, var_theta]
        # — a compact belief representation for RL input.
        
        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        ts = np.array([p.theta for p in self.particles])
        return np.array([
            xs.mean(), ys.mean(), ts.mean(),
            xs.var(), ys.var(), ts.var(),
        ])

    def get_epistemic_uncertainty(self) -> float:
        # Scalar uncertainty: entropy of particle distribution.
        # High entropy = high epistemic uncertainty about robot state.
        

        weights = np.array([p.weight for p in self.particles])
        weights = weights / weights.sum()
        entropy = -np.sum(weights * np.log(weights + 1e-300))
        return float(entropy / np.log(self.n))  # normalized [0, 1]

    def get_pose_estimate(self) -> tuple:
        # Best estimate: weighted mean

        xs = np.array([p.x for p in self.particles])
        ys = np.array([p.y for p in self.particles])
        ts = np.array([p.theta for p in self.particles])
        ws = np.array([p.weight for p in self.particles])
        ws /= ws.sum()
        return float((xs * ws).sum()), float((ys * ws).sum()), float((ts * ws).sum())
