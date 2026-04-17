# ROS2 node: Particle filter localization for TurtleBot3.


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import math
from scipy.ndimage import distance_transform_edt

from uncertainty_nav.particle_filter import ParticleFilter


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__("particle_filter_node")

        self.declare_parameter("n_particles", 500)
        self.declare_parameter("map_size", 10.0)
        self.declare_parameter("n_scan_beams_used", 36)  # subsample for speed
        self.declare_parameter("obs_noise_std", 0.2)

        n_particles        = self.get_parameter("n_particles").value
        map_size           = self.get_parameter("map_size").value
        self.n_beams_used  = self.get_parameter("n_scan_beams_used").value
        obs_noise_std      = self.get_parameter("obs_noise_std").value
        half               = map_size / 2.0

        self.pf = ParticleFilter(
            n_particles=n_particles,
            map_bounds=(-half, half, -half, half),
            obs_noise_std=obs_noise_std,
        )

        # Occupancy grid state
        self._occ_grid: np.ndarray | None = None   # 2D bool array: True = occupied
        self._grid_res: float = 0.05               # m/cell (default, updated from map)
        self._grid_origin_x: float = -half
        self._grid_origin_y: float = -half
        self._dist_map: np.ndarray | None = None   # EDT distance map in cells

        self._last_odom = None
        self._last_time: float | None = None

        #  QoS 
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        #  Subscribers 
        self.create_subscription(LaserScan,     "/scan", self._scan_cb, sensor_qos)
        self.create_subscription(Odometry,      "/odom", self._odom_cb, 10)
        self.create_subscription(OccupancyGrid, "/map",  self._map_cb,  map_qos)

        #  Publishers
        self._belief_pub   = self.create_publisher(Float32MultiArray, "/belief_state",   10)
        self._unc_pub      = self.create_publisher(Float32,           "/pf_uncertainty", 10)
        self._particle_pub = self.create_publisher(PoseArray,         "/particle_cloud", 10)

        self.get_logger().info(
            f"ParticleFilterNode ready | N={n_particles} | "
            f"beams_used={self.n_beams_used}"
        )

    #  Map callback

    def _map_cb(self, msg: OccupancyGrid):        
        # Build occupancy grid and distance transform map.
        
        self._grid_res      = msg.info.resolution
        self._grid_origin_x = msg.info.origin.position.x
        self._grid_origin_y = msg.info.origin.position.y
        width  = msg.info.width
        height = msg.info.height

        # OccupancyGrid data: 0=free, 100=occupied, -1=unknown
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        # Treat occupied (≥50) and unknown (-1) as obstacles for raycast
        self._occ_grid = (data >= 50) | (data == -1)

        # Distance transform: distance from each cell to nearest obstacle (in cells)
        # Used for fast likelihood computation
        free = ~self._occ_grid
        self._dist_map = distance_transform_edt(free).astype(np.float32) * self._grid_res

        self.get_logger().info(
            f"Map received: {width}×{height} @ {self._grid_res:.3f} m/cell"
        )

    #  Odometry callback 

    def _odom_cb(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0.0:
                v     = msg.twist.twist.linear.x
                omega = msg.twist.twist.angular.z
                self.pf.predict(v, omega, dt)
        self._last_odom = msg
        self._last_time = now

    #  Scan callback 

    def _scan_cb(self, msg: LaserScan):
        # Update particle weights using real LDS-01 scan.
        # Uses occupancy-grid raycast if map is available,
        # falls back to wall-only raycast otherwise.
        
        ranges_full = np.array(msg.ranges, dtype=np.float32)
        ranges_full = np.where(np.isfinite(ranges_full), ranges_full, msg.range_max)
        ranges_full = np.clip(ranges_full, msg.range_min, msg.range_max)

        # Subsample beams for computational efficiency
        n_total = len(ranges_full)
        indices = np.linspace(0, n_total - 1, self.n_beams_used, dtype=int)
        obs_ranges = ranges_full[indices]

        # Beam angles for subsampled beams
        angle_min  = msg.angle_min
        angle_incr = msg.angle_increment
        all_angles = angle_min + np.arange(n_total) * angle_incr
        beam_angles = all_angles[indices]  # relative to robot heading

        if self._dist_map is not None:
            expected_fn = self._make_grid_raycast_fn(
                beam_angles, msg.range_max, msg.range_min
            )
        else:
            expected_fn = self._make_wall_raycast_fn(
                beam_angles, msg.range_max
            )

        self.pf.update(obs_ranges, expected_fn)

        #  Publish
        belief = self.pf.get_belief_state()
        uncertainty = self.pf.get_epistemic_uncertainty()

        belief_msg = Float32MultiArray()
        belief_msg.data = belief.tolist()
        self._belief_pub.publish(belief_msg)
        self._unc_pub.publish(Float32(data=float(uncertainty)))
        self._publish_particles()

    #  Raycast functions
    def _make_grid_raycast_fn(self, beam_angles: np.ndarray, range_max: float, range_min: float):
        """
        Returns expected_ranges_fn(x, y, theta) using occupancy grid EDT.

        For each beam: step along the beam direction in the distance map.
        The EDT value at each cell = distance to nearest obstacle.
        We find the first cell where EDT < step_size (i.e., we hit an obstacle).

        This is a vectorized approximation — fast enough for 500 particles × 36 beams.
        """
        dist_map   = self._dist_map
        occ_grid   = self._occ_grid
        res        = self._grid_res
        origin_x   = self._grid_origin_x
        origin_y   = self._grid_origin_y
        height, width = occ_grid.shape
        step       = res * 0.8  # step slightly smaller than cell size

        def expected_ranges_fn(px: float, py: float, ptheta: float) -> np.ndarray:
            expected = np.full(len(beam_angles), range_max, dtype=np.float32)
            for i, rel_angle in enumerate(beam_angles):
                angle = ptheta + rel_angle
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                r = range_min
                while r < range_max:
                    wx = px + r * cos_a
                    wy = py + r * sin_a
                    # World → grid cell
                    cx = int((wx - origin_x) / res)
                    cy = int((wy - origin_y) / res)
                    if cx < 0 or cx >= width or cy < 0 or cy >= height:
                        expected[i] = r
                        break
                    if occ_grid[cy, cx]:
                        expected[i] = r
                        break
                    # Use EDT to skip ahead: jump by (edt_value - step) safely
                    edt_val = dist_map[cy, cx]
                    r += max(step, edt_val - step)
                else:
                    expected[i] = range_max
            return expected

        return expected_ranges_fn

    def _make_wall_raycast_fn(self, beam_angles: np.ndarray, range_max: float):
        # Fallback raycast against arena boundary walls only (no map needed).
        # Used before the map is received.
        
        half = 5.0  # default arena half-size

        def expected_ranges_fn(px: float, py: float, ptheta: float) -> np.ndarray:
            expected = np.full(len(beam_angles), range_max, dtype=np.float32)
            for i, rel_angle in enumerate(beam_angles):
                angle = ptheta + rel_angle
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                min_dist = range_max
                # Intersect with 4 walls
                for wall, coord, component in [
                    ( half, px, cos_a), (-half, px, cos_a),
                    ( half, py, sin_a), (-half, py, sin_a),
                ]:
                    if abs(component) > 1e-9:
                        d = (wall - coord) / component
                        if 0.0 < d < min_dist:
                            min_dist = d
                expected[i] = min_dist
            return expected

        return expected_ranges_fn

    #  Publish particles

    def _publish_particles(self):
        msg = PoseArray()
        msg.header.frame_id = "odom"
        msg.header.stamp    = self.get_clock().now().to_msg()
        for p in self.pf.particles:
            pose = Pose()
            pose.position.x    = p.x
            pose.position.y    = p.y
            pose.orientation.z = math.sin(p.theta / 2.0)
            pose.orientation.w = math.cos(p.theta / 2.0)
            msg.poses.append(pose)
        self._particle_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
