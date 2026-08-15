#!/usr/bin/env python3
import copy
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class EmptyGasMapNode(Node):
    def __init__(self):
        super().__init__('gas_empty_map_node')
        defaults = {
            'source_x': 10.0, 'source_y': 10.0, 'source_strength': 1.0,
            'source_sigma': 5.0,
            'map_min_x': 0.0, 'map_max_x': 50.0,
            'map_min_y': 0.0, 'map_max_y': 50.0,
            'nav_resolution': 0.05, 'ground_truth_resolution': 0.1,
            'gmrf_resolution': 1.0, 'map_margin': 2.5,
            'log_concentration_min': -4.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        if min(self.source_sigma, self.nav_resolution,
               self.ground_truth_resolution, self.gmrf_resolution) <= 0.0:
            raise ValueError('sigma and grid resolutions must be positive')
        if self.log_concentration_min >= 0.0:
            raise ValueError('log_concentration_min must be negative')

        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map', transient_qos())
        self.gt_pub = self.create_publisher(
            OccupancyGrid, '/gas_lrs/ground_truth', transient_qos())
        self.gt_log_pub = self.create_publisher(
            OccupancyGrid, '/gas_lrs/ground_truth_log', transient_qos())
        self.gmrf_grid_pub = self.create_publisher(
            OccupancyGrid, '/gas_lrs/gmrf_domain', transient_qos())
        self.source_pub = self.create_publisher(
            Marker, '/gas_lrs/source', transient_qos())
        self.sensor_pub = self.create_publisher(
            Float64, '/gas_lrs/sensor_concentration', 20)
        self.create_subscription(PoseStamped, '/gas_sensor/sensor_pose', self.pose_callback, 20)
        self.initialize_maps()

    def concentration(self, x, y):
        distance_sq = (x - self.source_x) ** 2 + (y - self.source_y) ** 2
        return min(1.0, max(0.0, self.source_strength * math.exp(
            -distance_sq / (2.0 * self.source_sigma ** 2))))

    def pose_callback(self, msg):
        self.sensor_pub.publish(Float64(data=self.concentration(
            msg.pose.position.x, msg.pose.position.y)))

    def make_grid(self, resolution, min_x, min_y, max_x, max_y, occupied_border=False):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(resolution)
        msg.info.width = int(round((max_x - min_x) / resolution))
        msg.info.height = int(round((max_y - min_y) / resolution))
        msg.info.origin.position.x = float(min_x)
        msg.info.origin.position.y = float(min_y)
        msg.info.origin.orientation.w = 1.0
        data = [0] * (msg.info.width * msg.info.height)
        if occupied_border:
            border = max(1, int(math.ceil(0.1 / resolution)))
            for row in range(msg.info.height):
                offset = row * msg.info.width
                if row < border or row >= msg.info.height - border:
                    data[offset:offset + msg.info.width] = [100] * msg.info.width
                else:
                    data[offset:offset + border] = [100] * border
                    data[offset + msg.info.width - border:offset + msg.info.width] = [100] * border
        msg.data = data
        return msg

    def initialize_maps(self):
        nav_map = self.make_grid(
            self.nav_resolution,
            self.map_min_x - self.map_margin, self.map_min_y - self.map_margin,
            self.map_max_x + self.map_margin, self.map_max_y + self.map_margin,
            occupied_border=True)
        gmrf_map = self.make_grid(
            self.gmrf_resolution,
            self.map_min_x - 0.5 * self.gmrf_resolution,
            self.map_min_y - 0.5 * self.gmrf_resolution,
            self.map_max_x + 0.5 * self.gmrf_resolution,
            self.map_max_y + 0.5 * self.gmrf_resolution)
        self.map_pub.publish(nav_map)
        self.gmrf_grid_pub.publish(gmrf_map)
        self.publish_ground_truth()
        self.publish_source()
        self.get_logger().info(
            f'Empty gas maps ready: nav={self.nav_resolution:.3f}m, '
            f'GT={self.ground_truth_resolution:.3f}m, '
            f'GMRF={self.gmrf_resolution:.3f}m')

    def publish_ground_truth(self):
        half_cell = 0.5 * self.ground_truth_resolution
        output = self.make_grid(
            self.ground_truth_resolution,
            self.map_min_x - half_cell, self.map_min_y - half_cell,
            self.map_max_x + half_cell, self.map_max_y + half_cell)
        linear_values = []
        log_values = []
        log_min = float(self.log_concentration_min)
        concentration_floor = math.exp(log_min)
        for row in range(output.info.height):
            y = output.info.origin.position.y + (row + 0.5) * output.info.resolution
            for col in range(output.info.width):
                x = output.info.origin.position.x + (col + 0.5) * output.info.resolution
                concentration = self.concentration(x, y)
                linear_values.append(int(round(100.0 * concentration)))
                log_concentration = math.log(max(concentration, concentration_floor))
                log_normalized = min(1.0, max(
                    0.0, (log_concentration - log_min) / -log_min))
                log_values.append(int(round(100.0 * log_normalized)))
        output.data = linear_values
        self.gt_pub.publish(output)
        log_output = copy.deepcopy(output)
        log_output.data = log_values
        self.gt_log_pub.publish(log_output)

    def publish_source(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_lrs_source'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.source_x
        marker.pose.position.y = self.source_y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.4
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.1, 0.0, 1.0
        self.source_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = EmptyGasMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
