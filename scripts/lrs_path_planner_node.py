#!/usr/bin/env python3
import copy
import math

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker

from lrs_tsp_planner import LrsPoint, solve_lrs_tsp


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class LrsPathPlannerNode(Node):
    def __init__(self):
        super().__init__('lrs_path_planner_node')
        defaults = {
            'lrs_spacing': 10.0,
            'lrs_min_x': 0.0, 'lrs_max_x': 50.0,
            'lrs_min_y': 0.0, 'lrs_max_y': 50.0,
            'robot_start_x': 0.0, 'robot_start_y': 0.0,
            'goal_clearance': 0.25, 'tsp_time_limit': 60.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        if self.lrs_spacing <= 0.0 or self.goal_clearance < 0.0:
            raise ValueError('LRS spacing must be positive and clearance non-negative')

        self.path_pub = self.create_publisher(
            Path, '/gas_lrs/tsp_route', transient_qos())
        self.points_pub = self.create_publisher(
            Marker, '/gas_lrs/sampling_status', transient_qos())
        self.points_log_pub = self.create_publisher(
            Marker, '/gas_lrs/sampling_status_log', transient_qos())
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, transient_qos())
        self.planned = False

    def map_callback(self, msg):
        if self.planned:
            return
        points = self.build_points(msg)
        expected_rows = int(round(
            (self.lrs_max_y - self.lrs_min_y) / self.lrs_spacing)) + 1
        expected_cols = int(round(
            (self.lrs_max_x - self.lrs_min_x) / self.lrs_spacing)) + 1
        expected_count = expected_rows * expected_cols
        if len(points) != expected_count:
            raise RuntimeError(
                f'LRS domain expected {expected_count} clear points, '
                f'found {len(points)}')

        result = solve_lrs_tsp(
            points, (self.robot_start_x, self.robot_start_y),
            self.tsp_time_limit)
        ordered = [points[index] for index in result.tour]
        self.publish_route(ordered)
        self.publish_points(ordered)
        self.planned = True
        self.get_logger().info(
            f'LRS P1 complete: points={len(points)}, status={result.status}, '
            f'objective={result.objective:.3f}m, gap={result.gap:.6f}, '
            f'cuts={result.cuts}')

    def is_clear(self, msg, x, y):
        radius = int(math.ceil(self.goal_clearance / msg.info.resolution))
        col = int((x - msg.info.origin.position.x) / msg.info.resolution)
        row = int((y - msg.info.origin.position.y) / msg.info.resolution)
        for rr in range(row - radius, row + radius + 1):
            for cc in range(col - radius, col + radius + 1):
                if (rr < 0 or cc < 0 or rr >= msg.info.height or
                        cc >= msg.info.width):
                    return False
                if msg.data[rr * msg.info.width + cc] != 0:
                    return False
        return True

    def build_points(self, msg):
        points = []
        row = 0
        y = self.lrs_min_y
        while y <= self.lrs_max_y + 1.0e-9:
            col = 0
            x = self.lrs_min_x
            while x <= self.lrs_max_x + 1.0e-9:
                if self.is_clear(msg, x, y):
                    points.append(LrsPoint(row, col, x, y))
                x += self.lrs_spacing
                col += 1
            y += self.lrs_spacing
            row += 1
        return points

    def publish_route(self, ordered):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for point in ordered + ordered[:1]:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def publish_points(self, ordered):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_lrs_sampling_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.18
        marker.color.r = marker.color.g = marker.color.b = 0.55
        marker.color.a = 1.0
        marker.points = [Point(x=p.x, y=p.y, z=0.08) for p in ordered]
        self.points_pub.publish(marker)
        log_marker = copy.deepcopy(marker)
        log_marker.ns = 'gas_lrs_sampling_points_log'
        self.points_log_pub.publish(log_marker)


def main(args=None):
    rclpy.init(args=args)
    node = LrsPathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
