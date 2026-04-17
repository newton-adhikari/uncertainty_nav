# ROS2 node: RViz uncertainty visualization.

import rclpy
from rclpy.node import Node

class RVizUncertaintyNode(Node):
    def __init__(self):
        super().__init__("rviz_uncertainty_node")

        self.get_logger().info("RVizUncertaintyNode started")

def main(args=None):
    rclpy.init(args=args)
    node = RVizUncertaintyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
