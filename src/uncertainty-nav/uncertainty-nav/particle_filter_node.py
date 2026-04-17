# ROS2 node: Particle filter localization for TurtleBot3.


import rclpy
from rclpy.node import Node
from uncertainty_nav.particle_filter import ParticleFilter


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__("particle_filter_node")

        self.declare_parameter("n_particles", 500)
        self.declare_parameter("n_scan_beams_used", 36)  # subsample for speed

        n_particles        = self.get_parameter("n_particles").value
        self.n_beams_used  = self.get_parameter("n_scan_beams_used").value

        self.get_logger().info(
            f"ParticleFilterNode ready | N={n_particles} | "
            f"beams_used={self.n_beams_used}"
        )

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
