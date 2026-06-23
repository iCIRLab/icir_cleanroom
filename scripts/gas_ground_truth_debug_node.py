#!/usr/bin/env python3
"""
DEBUG ONLY: 이 노드는 gas_source(들)의 실제(ground-truth) 위치/농도를 직접 사용하여
맵 전체의 가스 분포를 계산해서 시각화합니다.
실제 로봇/센서 환경에는 존재하지 않는 정보(소스의 진짜 위치)를 사용하므로,
patrol/seeking 알고리즘의 입력으로 사용하면 안 되며, 개발 중 디버깅 시각화 용도로만 씁니다.

여러 개의 gas_source가 있어도 동작합니다 - "/.../concentration" 형태의 토픽 이름 중
"gas_source"가 포함된 것을 주기적으로 스캔해서 자동으로 구독을 추가합니다.
"""

import math
import re

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

SOURCE_TOPIC_PATTERN = re.compile(r'^/?(.*gas_source.*)/concentration$')


class GasGroundTruthDebugNode(Node):
    def __init__(self):
        super().__init__('gas_ground_truth_debug_node')

        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_info = None

        # source_name -> {'pos': (x, y) or None, 'concentration': float}
        self.sources = {}
        self.concentration_subs = {}
        self.pose_subs = {}

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)

        self.debug_map_pub = self.create_publisher(OccupancyGrid, '/debug/gas_concentration_map', map_qos)

        self.create_timer(1.0, self.discover_sources)
        self.create_timer(0.5, self.publish_debug_map)  # 2Hz

    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info

    def discover_sources(self):
        for topic_name, _ in self.get_topic_names_and_types():
            match = SOURCE_TOPIC_PATTERN.match(topic_name)
            if not match:
                continue

            source_name = match.group(1)
            if source_name in self.sources:
                continue

            self.get_logger().info(f'새 gas_source 발견: {source_name}')
            self.sources[source_name] = {'pos': None, 'concentration': 0.0}

            concentration_topic = f'/{source_name}/concentration'
            pose_topic = f'/{source_name}/pose'

            self.concentration_subs[source_name] = self.create_subscription(
                Float64, concentration_topic,
                self._make_concentration_callback(source_name), 10)

            self.pose_subs[source_name] = self.create_subscription(
                PoseStamped, pose_topic,
                self._make_pose_callback(source_name), 10)

    def _make_concentration_callback(self, source_name):
        def callback(msg: Float64):
            self.sources[source_name]['concentration'] = msg.data
        return callback

    def _make_pose_callback(self, source_name):
        def callback(msg: PoseStamped):
            self.sources[source_name]['pos'] = (msg.pose.position.x, msg.pose.position.y)
        return callback

    def publish_debug_map(self):
        if self.map_info is None:
            return

        active_sources = [s for s in self.sources.values() if s['pos'] is not None]
        if not active_sources:
            return

        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        data = [0] * (width * height)

        for row in range(height):
            wy = origin_y + (row + 0.5) * resolution
            for col in range(width):
                wx = origin_x + (col + 0.5) * resolution

                total = 0.0
                for src in active_sources:
                    sx, sy = src['pos']
                    distance = math.hypot(wx - sx, wy - sy)
                    distance = max(distance, 1e-3)

                    attenuation = min(1.0 / (distance * distance), 1.0)
                    total += src['concentration'] * attenuation

                value = max(0.0, min(total, 100.0))
                data[row * width + col] = int(value)

        out_msg = OccupancyGrid()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'map'
        out_msg.info = self.map_info
        out_msg.data = data

        self.debug_map_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GasGroundTruthDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
