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
            'lrs_spacing': 10.0,    # LRS(격자) 샘플링 지점들 사이의 간격(m)
            'lrs_min_x': 0.0,       # 격자 생성 범위 x축 최소(m)
            'lrs_max_x': 50.0,      # 격자 생성 범위 x축 최대(m)
            'lrs_min_y': 0.0,       # 격자 생성 범위 y축 최소(m)
            'lrs_max_y': 50.0,      # 격자 생성 범위 y축 최대(m)
            'robot_start_x': 0.0,   # TSP 경로 계산 시 로봇 출발 x 위치
            'robot_start_y': 0.0,   # TSP 경로 계산 시 로봇 출발 y 위치
            'goal_clearance': 0.25, # 후보 지점이 장애물과 유지해야할 최소 거리(m)
            'tsp_time_limit': 60.0, # TSP 최적 경로 계산 제한 시간(초)
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        if self.lrs_spacing <= 0.0 or self.goal_clearance < 0.0:
            raise ValueError('LRS spacing must be positive and clearance non-negative')

        self.path_pub = self.create_publisher(
            Path, '/gas_mapping/lrs/route', transient_qos())
        self.points_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/status', transient_qos())
        self.points_log_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/status_log', transient_qos())
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
        marker.ns = 'gas_mapping_lrs_status'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.18
        marker.color.r = marker.color.g = marker.color.b = 0.55
        marker.color.a = 1.0
        marker.points = [Point(x=p.x, y=p.y, z=0.08) for p in ordered]
        self.points_pub.publish(marker)
        log_marker = copy.deepcopy(marker)
        log_marker.ns = 'gas_mapping_lrs_status_log'
        self.points_log_pub.publish(log_marker)


def main(args=None):
    rclpy.init(args=args)
    node = LrsPathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
