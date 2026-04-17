import rclpy
from rclpy.node import Node

class UncertaintyAgentNode(Node):
    def __init__(self):
        super().__init__("uncertainty_agent")
