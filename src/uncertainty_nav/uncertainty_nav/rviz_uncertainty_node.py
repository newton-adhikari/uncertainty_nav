# ROS2 node: RViz uncertainty visualization.

"""
Subscribes: /uncertainty (Float32), /pf_uncertainty (Float32)
Publishes:  /uncertainty_heatmap (MarkerArray) — colored grid overlay
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np


class RVizUncertaintyNode(Node):
    def __init__(self):
        super().__init__("rviz_uncertainty_node")

        self._ensemble_unc = 0.0
        self._pf_unc = 0.0

        self.create_subscription(Float32, "/uncertainty", self._ens_cb, 10)
        self.create_subscription(Float32, "/pf_uncertainty", self._pf_cb, 10)
        self._marker_pub = self.create_publisher(MarkerArray, "/uncertainty_heatmap", 10)

        self.create_timer(0.5, self._publish_heatmap)
        self.get_logger().info("RVizUncertaintyNode started")

    def _ens_cb(self, msg: Float32):
        self._ensemble_unc = msg.data

    def _pf_cb(self, msg: Float32):
        self._pf_unc = msg.data

    def _publish_heatmap(self):
        # Publish two sphere markers showing ensemble vs PF uncertainty
        array = MarkerArray()
        for i, (unc, label, x_offset) in enumerate([
            (self._ensemble_unc, "Ensemble", 0.0),
            (self._pf_unc, "PF", 1.5),
        ]):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = self.get_clock().now().to_msg()
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x_offset
            m.pose.position.y = -3.0
            m.pose.position.z = 0.5
            m.scale.x = m.scale.y = m.scale.z = float(0.3 + unc * 1.0)
            m.color = ColorRGBA(
                r=float(min(unc * 3.0, 1.0)),
                g=float(max(1.0 - unc * 3.0, 0.0)),
                b=0.0,
                a=0.8,
            )
            array.markers.append(m)

            # Text label
            t = Marker()
            t.header = m.header
            t.id = i + 10
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = x_offset
            t.pose.position.y = -3.0
            t.pose.position.z = 1.2
            t.scale.z = 0.2
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = f"{label}: {unc:.3f}"
            array.markers.append(t)

        self._marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = RVizUncertaintyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
