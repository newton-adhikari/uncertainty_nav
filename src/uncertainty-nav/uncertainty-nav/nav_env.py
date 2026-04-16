import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EnvConfig:
    # Map
    map_size: float = 10.0
    n_static_obstacles: int = 6       # reduced from 8 — less crowded for early learning

    # Sensor — TurtleBot3 LDS-01 real specs
    n_laser_beams: int = 36           # subsampled from 360 for training speed
    max_range: float = 3.5
    min_range: float = 0.12
    fov_deg: float = 360.0

    # Noise
    laser_noise_std: float = 0.01
    occlusion_prob: float = 0.0
    dropout_prob: float = 0.0

    # Task
    goal_radius: float = 0.4          # slightly larger — easier to reach
    collision_radius: float = 0.18    # TurtleBot3 Waffle Pi ≈ 0.178 m
    obstacle_radius: float = 0.3      # half-size of obstacle for collision check
    max_steps: int = 500
    dt: float = 0.1

    # TurtleBot3 Waffle Pi velocity limits
    max_linear_vel: float = 0.26
    max_angular_vel: float = 1.82

    # Reward shaping
    goal_reward: float = 100.0
    collision_penalty: float = -10.0  # reduced: less catastrophic, easier to learn from
    step_penalty: float = -0.05       # reduced: agent not punished too hard for exploring
    progress_reward_scale: float = 1.0  # reward for moving toward goal

    # Interior walls for perceptual aliasing (list of (x1,y1,x2,y2) segments)
    interior_walls: tuple = ()

    # Dynamic obstacles
    n_dynamic_obstacles: int = 0
    dynamic_speed: float = 0.15       # m/s


ENV_A = EnvConfig(
    laser_noise_std=0.05,
    occlusion_prob=0.05,
    fov_deg=180.0,
    n_static_obstacles=8,
    dropout_prob=0.0,
)

ENV_B = EnvConfig(
    map_size=12.0,                    
    laser_noise_std=0.12,             
    occlusion_prob=0.20,              
    fov_deg=120.0,                    
    n_static_obstacles=10,            # moderate clutter
    dropout_prob=0.08,                # occasional dropout
    max_steps=600,                    # enough steps for larger map
    
    # Interior walls create corridors and perceptual aliasing
    interior_walls=(
        (-2.0, -4.0, -2.0, 2.0),     # vertical wall left
        (2.0, -2.0, 2.0, 4.0),       # vertical wall right
    ),
    n_dynamic_obstacles=2,            # 2 moving obstacles
    dynamic_speed=0.10,               # slower dynamics
)

# --- Distribution shift spectrum (Pillar 2) ---

# Env C: Sensor-only shift — Env A layout + Env B sensor degradation
# Isolates the effect of sensor noise/occlusion from layout change
ENV_C = EnvConfig(
    map_size=10.0,                    # same as Env A
    laser_noise_std=0.12,             # Env B noise
    occlusion_prob=0.20,              # Env B occlusion
    fov_deg=120.0,                    # Env B FoV
    n_static_obstacles=8,             # same as Env A
    dropout_prob=0.08,                # Env B dropout
    max_steps=500,                    # same as Env A
    interior_walls=(),                # no interior walls (Env A)
    n_dynamic_obstacles=0,            # no dynamic obstacles (Env A)
)

# Env D: Layout-only shift — Env B layout + Env A sensor parameters
# Isolates the effect of structural novelty from sensor degradation
ENV_D = EnvConfig(
    map_size=12.0,                    # Env B size
    laser_noise_std=0.05,             # Env A noise
    occlusion_prob=0.05,              # Env A occlusion
    fov_deg=180.0,                    # Env A FoV
    n_static_obstacles=10,            # Env B obstacles
    dropout_prob=0.0,                 # no dropout (Env A)
    max_steps=600,                    # Env B steps
    interior_walls=(                  # Env B walls
        (-2.0, -4.0, -2.0, 2.0),
        (2.0, -2.0, 2.0, 4.0),
    ),
    n_dynamic_obstacles=2,            # Env B dynamics
    dynamic_speed=0.10,
)

class PartialObsNavEnv(gym.Env):
    def __init__(self, config: EnvConfig = ENV_A, seed: Optional[int] = None):
        super().__init__()
        self.cfg = config
        self.rng = np.random.default_rng(seed)

        obs_dim = config.n_laser_beams + 3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._obstacles = []
        self._dynamic_obstacles = []  # list of (pos, velocity) pairs
        self._step = 0
        self._robot_pose = np.zeros(3)
        self._goal = np.zeros(2)
        self._prev_dist_to_goal = 0.0
        self._episode_path_length = 0.0
        self._optimal_path_length = 0.0

        # Parse interior walls into segments
        self._interior_walls = []
        for wall in config.interior_walls:
            self._interior_walls.append(
                (np.array([wall[0], wall[1]]), np.array([wall[2], wall[3]]))
            )

        # Precompute beam angles (relative to robot heading = 0)
        self._beam_angles_rel = np.linspace(
            0, 2 * np.pi, config.n_laser_beams, endpoint=False
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step = 0
        self._episode_path_length = 0.0
        self._place_robot_and_goal()
        self._place_obstacles()
        self._place_dynamic_obstacles()
        self._prev_dist_to_goal = float(
            np.linalg.norm(self._goal - self._robot_pose[:2])
        )

        obs = self._get_obs()
        return obs.astype(np.float32), {"optimal_path_length": self._optimal_path_length}
    
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        v     = action[0] * self.cfg.max_linear_vel
        omega = action[1] * self.cfg.max_angular_vel

        prev_pos = self._robot_pose[:2].copy()
        self._update_robot(v, omega)
        self._update_dynamic_obstacles()

        dist_moved = float(np.linalg.norm(self._robot_pose[:2] - prev_pos))
        self._episode_path_length += dist_moved
        self._step += 1

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()
        truncated = self._step >= self.cfg.max_steps

        info.update({
            "path_length": self._episode_path_length,
            "optimal_path_length": self._optimal_path_length,
            "step": self._step,
        })
        return obs.astype(np.float32), reward, terminated, truncated, info
    
    def _place_robot_and_goal(self):
        half = self.cfg.map_size / 2 - 1.0   # keep 1 m from walls
        for _ in range(1000):
            robot_xy = self.rng.uniform(-half, half, size=2)
            goal_xy  = self.rng.uniform(-half, half, size=2)
            dist = float(np.linalg.norm(goal_xy - robot_xy))
            if 2.5 <= dist <= half * 1.8:     # not too close, not too far
                break
        self._robot_pose = np.array([robot_xy[0], robot_xy[1],
                                     self.rng.uniform(-np.pi, np.pi)])
        self._goal = goal_xy
        self._optimal_path_length = dist

    def _place_obstacles(self):
        
        # Place obstacles with guaranteed clearance from robot AND goal.
        # Clearance = collision_radius + obstacle_radius + 0.5 m safety margin.
        
        half = self.cfg.map_size / 2 - 1.0
        min_clear = self.cfg.collision_radius + self.cfg.obstacle_radius + 0.5

        self._obstacles = []
        for _ in range(self.cfg.n_static_obstacles):
            for attempt in range(500):
                pos = self.rng.uniform(-half, half, size=2)
                too_close_robot = np.linalg.norm(pos - self._robot_pose[:2]) < min_clear
                too_close_goal  = np.linalg.norm(pos - self._goal) < min_clear
                too_close_obs   = any(
                    np.linalg.norm(pos - o) < self.cfg.obstacle_radius * 2 + 0.2
                    for o in self._obstacles
                )
                if not too_close_robot and not too_close_goal and not too_close_obs:
                    self._obstacles.append(pos)
                    break

    def _place_dynamic_obstacles(self):
        # Place dynamic obstacles with random initial positions and velocities

        half = self.cfg.map_size / 2 - 1.5
        min_clear = self.cfg.collision_radius + self.cfg.obstacle_radius + 1.0
        self._dynamic_obstacles = []
        for _ in range(self.cfg.n_dynamic_obstacles):
            for _ in range(500):
                pos = self.rng.uniform(-half, half, size=2)
                if (np.linalg.norm(pos - self._robot_pose[:2]) > min_clear and
                        np.linalg.norm(pos - self._goal) > min_clear):
                    angle = self.rng.uniform(0, 2 * np.pi)
                    vel = np.array([np.cos(angle), np.sin(angle)]) * self.cfg.dynamic_speed
                    self._dynamic_obstacles.append([pos.copy(), vel.copy()])
                    break

    def _update_dynamic_obstacles(self):
        # Move dynamic obstacles, bounce off walls.
        half = self.cfg.map_size / 2 - self.cfg.obstacle_radius
        for dyn in self._dynamic_obstacles:
            dyn[0] += dyn[1] * self.cfg.dt
            # Bounce off walls
            for dim in range(2):
                if dyn[0][dim] > half:
                    dyn[0][dim] = half
                    dyn[1][dim] *= -1
                elif dyn[0][dim] < -half:
                    dyn[0][dim] = -half
                    dyn[1][dim] *= -1


    # robot's physics
    def _update_robot(self, v: float, omega: float):
        x, y, th = self._robot_pose
        x  += v * np.cos(th) * self.cfg.dt
        y  += v * np.sin(th) * self.cfg.dt
        th += omega * self.cfg.dt
        half = self.cfg.map_size / 2 - self.cfg.collision_radius
        x = np.clip(x, -half, half)
        y = np.clip(y, -half, half)
        self._robot_pose = np.array([x, y, th])

    # for faster training using vectorized raycast
    def _get_laser_scans(self) -> np.ndarray:
        
        #Vectorized laser scan simulation.
        #Uses numpy broadcasting instead of Python loops over beams.
        #50x faster than the previous per-beam loop for 36 beams.
        
        # Full dropout
        if self.rng.random() < self.cfg.dropout_prob:
            return np.ones(self.cfg.n_laser_beams, dtype=np.float32)

        x, y, th = self._robot_pose
        angles = self._beam_angles_rel + th   # (n_beams,) absolute angles

        # FoV mask
        if self.cfg.fov_deg < 360.0:
            half_fov = np.deg2rad(self.cfg.fov_deg) / 2.0
            rel = (self._beam_angles_rel + np.pi) % (2 * np.pi) - np.pi
            fov_mask = np.abs(rel) <= half_fov
        else:
            fov_mask = np.ones(self.cfg.n_laser_beams, dtype=bool)

        scans = np.full(self.cfg.n_laser_beams, self.cfg.max_range)

        # Wall intersections (vectorized)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        half_map = self.cfg.map_size / 2.0

        for wall, coord, component in [
            ( half_map, x, cos_a), (-half_map, x, cos_a),
            ( half_map, y, sin_a), (-half_map, y, sin_a),
        ]:
            with np.errstate(divide="ignore", invalid="ignore"):
                d = (wall - coord) / component
            valid = np.isfinite(d) & (d > 1e-4)
            scans = np.where(valid & (d < scans), d, scans)

        # Obstacle intersections (vectorized per obstacle) 
        all_obstacles = list(self._obstacles) + [d[0] for d in self._dynamic_obstacles]
        for obs_pos in all_obstacles:
            dx = obs_pos[0] - x
            dy = obs_pos[1] - y
            # Project obstacle center onto each beam
            proj = dx * cos_a + dy * sin_a          # distance along beam
            # Perpendicular distance from beam to obstacle center
            perp = np.abs(dx * sin_a - dy * cos_a)
            r = self.cfg.obstacle_radius
            hit = (proj > 0) & (perp < r)
            # Distance to hit point (chord entry)
            chord = np.where(hit, proj - np.sqrt(np.maximum(r**2 - perp**2, 0)), self.cfg.max_range)
            scans = np.where(hit & (chord < scans), chord, scans)

        # Interior wall intersections 
        for wall_start, wall_end in self._interior_walls:
            # Ray-segment intersection for each beam
            wx = wall_end[0] - wall_start[0]
            wy = wall_end[1] - wall_start[1]
            for i in range(self.cfg.n_laser_beams):
                dx_r = cos_a[i]
                dy_r = sin_a[i]
                denom = dx_r * wy - dy_r * wx
                if abs(denom) < 1e-10:
                    continue
                t = ((wall_start[0] - x) * wy - (wall_start[1] - y) * wx) / denom
                u = ((wall_start[0] - x) * dy_r - (wall_start[1] - y) * dx_r) / denom
                if t > 1e-4 and 0.0 <= u <= 1.0 and t < scans[i]:
                    scans[i] = t

        # Apply FoV mask, occlusion, noise
        scans = np.where(fov_mask, scans, self.cfg.max_range)

        if self.cfg.occlusion_prob > 0.0:
            occ = self.rng.random(self.cfg.n_laser_beams) < self.cfg.occlusion_prob
            scans = np.where(occ, self.cfg.max_range, scans)

        noise = self.rng.normal(0, self.cfg.laser_noise_std, self.cfg.n_laser_beams)
        scans = np.clip(scans + noise, self.cfg.min_range, self.cfg.max_range)

        return (scans / self.cfg.max_range).astype(np.float32)


    # robot's observations
    def _get_obs(self) -> np.ndarray:
        scans = self._get_laser_scans()
        dx = self._goal[0] - self._robot_pose[0]
        dy = self._goal[1] - self._robot_pose[1]
        dist = float(np.sqrt(dx**2 + dy**2))
        rel_angle = float(np.arctan2(dy, dx) - self._robot_pose[2])
        goal_info = np.array([
            np.cos(rel_angle),
            np.sin(rel_angle),
            np.clip(dist / self.cfg.map_size, 0.0, 1.0),
        ], dtype=np.float32)
        return np.concatenate([scans, goal_info])


    # main reward function
    def _compute_reward(self) -> Tuple[float, bool, dict]:
        x, y = self._robot_pose[:2]
        robot_pos = np.array([x, y])
        dist_to_goal = float(np.linalg.norm(self._goal - robot_pos))

        # Goal reached
        if dist_to_goal < self.cfg.goal_radius:
            self._prev_dist_to_goal = dist_to_goal
            return self.cfg.goal_reward, True, {"success": True, "collision": False}

        # Collision with obstacles (static + dynamic)
        all_obstacles = list(self._obstacles) + [d[0] for d in self._dynamic_obstacles]
        for obs_pos in all_obstacles:
            if np.linalg.norm(robot_pos - obs_pos) < (self.cfg.collision_radius + self.cfg.obstacle_radius):
                self._prev_dist_to_goal = dist_to_goal
                return self.cfg.collision_penalty, True, {"success": False, "collision": True}

        # Collision with interior walls
        for wall_start, wall_end in self._interior_walls:
            # Point-to-segment distance
            wall_vec = wall_end - wall_start
            wall_len_sq = np.dot(wall_vec, wall_vec)
            if wall_len_sq < 1e-10:
                continue
            t = np.clip(np.dot(robot_pos - wall_start, wall_vec) / wall_len_sq, 0.0, 1.0)
            closest = wall_start + t * wall_vec
            if np.linalg.norm(robot_pos - closest) < self.cfg.collision_radius + 0.05:
                self._prev_dist_to_goal = dist_to_goal
                return self.cfg.collision_penalty, True, {"success": False, "collision": True}

        # Progress reward: reward for getting closer to goal
        progress = self._prev_dist_to_goal - dist_to_goal
        self._prev_dist_to_goal = dist_to_goal

        reward = (self.cfg.step_penalty
                  + self.cfg.progress_reward_scale * progress)
        return float(reward), False, {"success": False, "collision": False}


    # metrics: NOTE: will use later
    def compute_spl(self) -> float:
        if self._optimal_path_length == 0:
            return 0.0
        return self._optimal_path_length / max(
            self._episode_path_length, self._optimal_path_length
        )

