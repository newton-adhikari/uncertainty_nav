# Uncertainty-aware navigation agent for TurtleBot3 Waffle Pi

"""
Partial observability simulation (applied in software on top of real laser):
  - FoV masking: zero out beams outside configured FoV (Env B = 90°)
  - Occlusion: randomly zero beams with probability occlusion_prob
  - Dropout: full scan zeroed with probability dropout_prob
  These are applied AFTER receiving the real /scan so the Gazebo physics
  still uses the full laser for collision, but the policy only sees the
  degraded observation
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped, PoseStamped as PS
from std_msgs.msg import Float32, ColorRGBA
from visualization_msgs.msg import Marker
import torch
import numpy as np
import os
import math
from tf_transformations import euler_from_quaternion

from uncertainty_nav.models import DeepEnsemble


# TurtleBot3 Waffle Pi hardware limits
TB3_MAX_LINEAR  = 0.26   # m/s
TB3_MAX_ANGULAR = 1.82   # rad/s
TB3_LASER_BEAMS = 360    # LDS-01 samples per scan
TB3_LASER_MAX_RANGE = 3.5  # m
TB3_LASER_MIN_RANGE = 0.12  # m


class UncertaintyAgentNode(Node):
    def __init__(self):
        super().__init__("uncertainty_agent")

        #  Parameters
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("n_members", 5)
        self.declare_parameter("hidden", 256)
        self.declare_parameter("n_laser_beams", 36)    # subsampled from 360 for policy
        self.declare_parameter("uncertainty_threshold", 0.3)
        self.declare_parameter("caution_scale", 0.5)
        # Partial observability applied in software (for Env B degradation)
        self.declare_parameter("fov_deg", 360.0)       # 360=Env A, 90=Env B
        self.declare_parameter("occlusion_prob", 0.0)  # 0.0=Env A, 0.2=Env B
        self.declare_parameter("dropout_prob", 0.0)    # 0.0=Env A, 0.05=Env B
        self.declare_parameter("map_size", 10.0)

        checkpoint      = self.get_parameter("checkpoint").value
        n_members       = self.get_parameter("n_members").value
        hidden          = self.get_parameter("hidden").value
        self.n_beams    = self.get_parameter("n_laser_beams").value
        self.unc_thresh = self.get_parameter("uncertainty_threshold").value
        self.caution    = self.get_parameter("caution_scale").value
        self.fov_deg    = self.get_parameter("fov_deg").value
        self.occ_prob   = self.get_parameter("occlusion_prob").value
        self.drop_prob  = self.get_parameter("dropout_prob").value
        self.map_size   = self.get_parameter("map_size").value

        # obs = n_beams laser scans + [cos(rel_angle), sin(rel_angle), dist_norm]
        obs_dim = self.n_beams + 3
        act_dim = 2
        self.device = torch.device("cpu")

        # Load ensemble from individual member checkpoints
        if checkpoint and os.path.exists(checkpoint):
            # If checkpoint points to a single file, try to find member files
            # e.g., checkpoints/ensemble_m0_policy.pt → look for m0..m{n-1}
            ckpt_dir = os.path.dirname(checkpoint) or "checkpoints"
            member_paths = [
                os.path.join(ckpt_dir, f"ensemble_m{i}_policy.pt")
                for i in range(n_members)
            ]
            existing = [p for p in member_paths if os.path.exists(p)]
            if existing:
                self.policy = DeepEnsemble.from_checkpoints(
                    existing, obs_dim, act_dim, hidden, device=self.device
                )
                self.get_logger().info(
                    f"Loaded ensemble with {len(existing)} members from {ckpt_dir}"
                )
            else:
                # Fallback: try loading as a single ensemble state dict
                self.policy = DeepEnsemble.from_checkpoints(
                    [checkpoint], obs_dim, act_dim, hidden, device=self.device
                )
                self.get_logger().warn(
                    f"No member files found; loaded single checkpoint: {checkpoint}"
                )
        else:
            # No checkpoint — create random ensemble for testing
            from uncertainty_nav.models import PolicyNetwork
            members = [PolicyNetwork(obs_dim, act_dim, hidden) for _ in range(n_members)]
            self.policy = DeepEnsemble(members).to(self.device)
            self.get_logger().warn("No checkpoint loaded — running with random weights")
        self.policy.eval()

        #  State
        self._latest_scan: np.ndarray | None = None
        self._robot_x     = 0.0
        self._robot_y     = 0.0
        self._robot_theta = 0.0
        self._goal_x      = 3.0
        self._goal_y      = 0.0
        self._goal_reached = False
        self._rng         = np.random.default_rng(42)

        #  QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        #  Subscribers
        self.create_subscription(LaserScan, "/scan", self._scan_cb, sensor_qos)
        self.create_subscription(Odometry,  "/odom", self._odom_cb, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_cb, 10)

        #  Publishers
        self._cmd_pub         = self.create_publisher(Twist,   "/cmd_vel",            10)
        self._unc_pub         = self.create_publisher(Float32, "/uncertainty",         10)
        self._marker_pub      = self.create_publisher(Marker,  "/uncertainty_marker",  10)
        self._raw_cmd_pub     = self.create_publisher(Twist,   "/cautious_cmd_vel",    10)
        self._path_pub        = self.create_publisher(Path,    "/robot_path",          10)
        self._goal_marker_pub = self.create_publisher(Marker,  "/goal_marker",         10)

        self._path_msg = Path()
        self._path_msg.header.frame_id = "odom"

        #  Control loop at 10 Hz (matches LDS-01 update rate)
        self.create_timer(0.1, self._control_loop)
        self.get_logger().info(
            f"UncertaintyAgentNode ready | "
            f"FoV={self.fov_deg}° | occ={self.occ_prob} | drop={self.drop_prob}"
        )

    #  Callbacks ─────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        
        # we Process LDS-01 scan in the following way:
        #  1. Replace inf/nan with max_range
        #  2. Clip to [min_range, max_range]
        #  3. Apply software partial observability (FoV mask, occlusion, dropout)
        #  4. Subsample to n_beams if needed
        #  5. Normalize to [0, 1]
        
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Replace invalid readings with max range
        ranges = np.where(np.isfinite(ranges), ranges, TB3_LASER_MAX_RANGE)
        ranges = np.clip(ranges, TB3_LASER_MIN_RANGE, TB3_LASER_MAX_RANGE)

        #  Software partial observability ────────────────────────────────
        # Full dropout (simulates complete sensor failure)
        if self._rng.random() < self.drop_prob:
            self._latest_scan = np.ones(self.n_beams, dtype=np.float32)
            return

        # FoV masking: zero out beams outside configured FoV
        # LDS-01 beams are indexed 0..359 = 0°..359°
        # We keep beams within ±fov/2 of the forward direction (beam 0 = forward)
        if self.fov_deg < 360.0:
            n_total = len(ranges)
            half_fov_beams = int((self.fov_deg / 360.0) * n_total / 2)
            mask = np.ones(n_total, dtype=bool)
            # Beams outside FoV → set to max range (unseen)
            mask[half_fov_beams: n_total - half_fov_beams] = False
            ranges[~mask] = TB3_LASER_MAX_RANGE

        # Per-beam occlusion
        if self.occ_prob > 0.0:
            occ_mask = self._rng.random(len(ranges)) < self.occ_prob
            ranges[occ_mask] = TB3_LASER_MAX_RANGE

        # Subsample to n_beams (if n_beams < 360)
        if len(ranges) != self.n_beams:
            indices = np.linspace(0, len(ranges) - 1, self.n_beams, dtype=int)
            ranges = ranges[indices]

        self._latest_scan = (ranges / TB3_LASER_MAX_RANGE).astype(np.float32)

    def _odom_cb(self, msg: Odometry):
        # Extract robot pose from odometry (diff drive plugin)
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._robot_theta = yaw

    def _goal_cb(self, msg: PoseStamped):
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        self._goal_reached = False

        # Clear path trail for new goal
        self._path_msg = Path()
        self._path_msg.header.frame_id = "odom"
        self.get_logger().info(f"New goal: ({self._goal_x:.2f}, {self._goal_y:.2f})")

    #  Control loop
    def _control_loop(self):
        if self._latest_scan is None:
            return

        # Check if goal is reached — stop and wait for new goal
        dx = self._goal_x - self._robot_x
        dy = self._goal_y - self._robot_y
        dist_to_goal = math.sqrt(dx * dx + dy * dy)

        if dist_to_goal < 0.4:  # goal_radius from nav_env.py
            cmd = Twist()  # all zeros = stop
            self._cmd_pub.publish(cmd)
            if not self._goal_reached:
                self._goal_reached = True
                self.get_logger().info(
                    f"Goal reached! dist={dist_to_goal:.2f}m. Waiting for new goal..."
                )
            return

        self._goal_reached = False
        obs = self._build_obs()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, uncertainty, is_cautious = self.policy.uncertainty_driven_action(
                obs_t,
                uncertainty_threshold=self.unc_thresh,
                caution_scale=self.caution,
            )

        action_np = action.squeeze(0).cpu().numpy()

        #  Publish raw (unscaled) cmd for logging
        raw_cmd = Twist()
        raw_cmd.linear.x  = float(np.clip(action_np[0] * TB3_MAX_LINEAR,  -TB3_MAX_LINEAR,  TB3_MAX_LINEAR))
        raw_cmd.angular.z = float(np.clip(action_np[1] * TB3_MAX_ANGULAR, -TB3_MAX_ANGULAR, TB3_MAX_ANGULAR))
        self._raw_cmd_pub.publish(raw_cmd)

        #  Publish actual cmd_vel (already scaled by uncertainty_driven_action)
        cmd = Twist()
        cmd.linear.x  = float(np.clip(action_np[0] * TB3_MAX_LINEAR,  -TB3_MAX_LINEAR,  TB3_MAX_LINEAR))
        cmd.angular.z = float(np.clip(action_np[1] * TB3_MAX_ANGULAR, -TB3_MAX_ANGULAR, TB3_MAX_ANGULAR))
        self._cmd_pub.publish(cmd)

        unc_val = float(uncertainty.item())
        self._unc_pub.publish(Float32(data=unc_val))
        self._publish_uncertainty_marker(unc_val, is_cautious)
        self._publish_path()
        self._publish_goal_marker()

        if is_cautious:
            self.get_logger().debug(
                f"CAUTIOUS | unc={unc_val:.3f} | "
                f"v={cmd.linear.x:.3f} | w={cmd.angular.z:.3f}"
            )

    def _build_obs(self) -> np.ndarray:
        # Build policy observation vector:
        # [scan_0, ..., scan_{n-1}, cos(rel_angle), sin(rel_angle), dist_norm]
        
        dx = self._goal_x - self._robot_x
        dy = self._goal_y - self._robot_y
        dist = math.sqrt(dx * dx + dy * dy)
        rel_angle = math.atan2(dy, dx) - self._robot_theta
        goal_info = np.array([
            math.cos(rel_angle),
            math.sin(rel_angle),
            float(np.clip(dist / self.map_size, 0.0, 1.0)),
        ], dtype=np.float32)
        return np.concatenate([self._latest_scan, goal_info])

    def _publish_uncertainty_marker(self, uncertainty: float, is_cautious: bool):
        # Sphere marker at robot position:
        # Size grows with uncertainty
        # Color: green (low) → red (high)
        # Visible in RViz as real-time uncertainty indicator
         
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns     = "uncertainty"
        marker.id     = 0
        marker.type   = Marker.SPHERE
        marker.action = Marker.ADD
        size = float(0.15 + uncertainty * 0.6)
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color = ColorRGBA(
            r=float(min(uncertainty * 3.0, 1.0)),
            g=float(max(1.0 - uncertainty * 3.0, 0.0)),
            b=0.0,
            a=0.75,
        )
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        self._marker_pub.publish(marker)


    def _publish_path(self):
        # Publish the robot's trajectory as a Path for RViz visualization
        pose = PoseStamped()
        pose.header.frame_id = "odom"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self._robot_x
        pose.pose.position.y = self._robot_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self._path_msg.poses.append(pose)
        self._path_msg.header.stamp = pose.header.stamp
        self._path_pub.publish(self._path_msg)

    def _publish_goal_marker(self):
        # Publish a green cylinder at the goal position in RViz
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self._goal_x
        marker.pose.position.y = self._goal_y
        marker.pose.position.z = 0.25
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4  # diameter = goal_radius * 2
        marker.scale.y = 0.4
        marker.scale.z = 0.5
        marker.color = ColorRGBA(r=0.0, g=0.9, b=0.2, a=0.7)
        self._goal_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = UncertaintyAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on shutdown
        stop = Twist()
        node._cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
