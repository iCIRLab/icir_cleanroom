#!/usr/bin/env python3
"""
가스 소스의 실제 위치(/gas_source/pose)와 농도(/gas_source/concentration)를
받아서 가우시안 확산 모델로 맵 전체에 가스 분포를 OccupancyGrid로 발행.
RViz에서 Map 타입으로 추가하면 가스가 어디서 발생해 어디까지 퍼지는지 확인 가능.
"""
import math
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory


DIFFUSION_SIGMA = 1.5   # 가스 확산 반경 (클수록 넓게 퍼짐, 단위: m)
PUBLISH_HZ      = 2.0   # 발행 주기


class GasSourceMapNode(Node):
    def __init__(self):
        super().__init__('gas_source_map_node')

        pkg_dir = get_package_share_directory('icir_cleanroom')
        map_yaml_path = pkg_dir + '/config/gas_attraction/map_data.yaml'
        with open(map_yaml_path) as f:
            cfg = yaml.safe_load(f)['map_data']

        self.origin_x   = cfg['map_origin_x']
        self.origin_y   = cfg['map_origin_y']
        self.resolution = cfg['map_resolution']
        self.max_intensity = cfg['max_intensity']
        self.width  = int(cfg['map_width']  / self.resolution)
        self.height = int(cfg['map_height'] / self.resolution)

        self.source_x = None
        self.source_y = None
        self.concentration = 0.0

        self.create_subscription(PoseStamped, '/gas_source/pose', self.pose_cb, 1)
        self.create_subscription(Float64, '/gas_source/concentration', self.conc_cb, 1)

        self.map_pub = self.create_publisher(OccupancyGrid, '/gas_source_map', 1)
        self.create_timer(1.0 / PUBLISH_HZ, self.publish_map)

        self.get_logger().info('gas_source_map_node started')

    def pose_cb(self, msg: PoseStamped):
        self.source_x = msg.pose.position.x
        self.source_y = msg.pose.position.y

    def conc_cb(self, msg: Float64):
        self.concentration = msg.data

    def publish_map(self):
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'map'
        grid.info.resolution = self.resolution
        grid.info.width  = self.width
        grid.info.height = self.height
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.orientation.w = 1.0

        # 소스 위치 미수신이거나 농도 0이면 빈 맵
        if self.source_x is None or self.concentration <= 0.0:
            grid.data = [0] * (self.width * self.height)
            self.map_pub.publish(grid)
            return

        data = []
        norm_conc = min(1.0, self.concentration / self.max_intensity)
        two_sigma2 = 2.0 * DIFFUSION_SIGMA * DIFFUSION_SIGMA

        for j in range(self.height):
            wy = self.origin_y + (j + 0.5) * self.resolution
            for i in range(self.width):
                wx = self.origin_x + (i + 0.5) * self.resolution
                dist2 = (wx - self.source_x) ** 2 + (wy - self.source_y) ** 2
                intensity = norm_conc * math.exp(-dist2 / two_sigma2)
                data.append(int(intensity * 100))

        grid.data = data
        self.map_pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = GasSourceMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
