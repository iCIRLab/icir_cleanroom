#!/usr/bin/env python3
"""
/gas_sensor/detected_concentration 값을 로봇 위에 떠다니는 텍스트로
RViz에 표시하기 위한 Marker(TEXT_VIEW_FACING)를 publish합니다.
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from visualization_msgs.msg import Marker


class GasConcentrationMarkerNode(Node):
    def __init__(self):
        super().__init__('gas_concentration_marker_node')

        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('height', 0.5)
        self.declare_parameter('text_scale', 0.2)

        self.frame_id = self.get_parameter('frame_id').value
        self.height = self.get_parameter('height').value
        self.text_scale = self.get_parameter('text_scale').value

        self.marker_pub = self.create_publisher(Marker, '/gas_sensor/concentration_marker', 10)

        self.create_subscription(
            Float64, '/gas_sensor/detected_concentration',
            self.concentration_callback, 10)

    def concentration_callback(self, msg: Float64):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_concentration'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = self.height

        marker.scale.z = self.text_scale
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.text = f'{msg.data:.2f}'

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GasConcentrationMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
