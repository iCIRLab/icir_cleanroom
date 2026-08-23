"""ROS adapter that publishes the low-resolution sampling route."""
import colorsys
import copy

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from ..mapping.grid_geometry import GridGeometry
from ..mapping.lrs_representatives import build_lrs_representatives
from ..mapping.path_distance import SamplingDistanceOracle
from ..planning.lrs_tsp import LrsPoint, solve_lrs_tsp


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class LrsPathPlannerNode(Node):
    def __init__(self):
        super().__init__('lrs_path_planner_node')
        defaults = {
            'robot_start_x': 0.0,   # TSP 경로 계산 시 로봇 출발 x 위치
            'robot_start_y': 0.0,   # TSP 경로 계산 시 로봇 출발 y 위치
            'tsp_time_limit': 60.0, # TSP 최적 경로 계산 제한 시간(초)
            'lrs_cluster_count': 36,
            'lrs_cluster_random_seed': 0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        if (isinstance(self.lrs_cluster_count, bool) or
                not isinstance(self.lrs_cluster_count, int) or
                self.lrs_cluster_count <= 0):
            raise ValueError(
                'lrs_cluster_count must be a positive integer')
        if (isinstance(self.lrs_cluster_random_seed, bool) or
                not isinstance(self.lrs_cluster_random_seed, int)):
            raise ValueError(
                'lrs_cluster_random_seed must be an integer')

        self.path_pub = self.create_publisher(
            Path, '/gas_mapping/lrs/route', transient_qos())
        self.points_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/status', transient_qos())
        self.points_log_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/status_log', transient_qos())
        self.clusters_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/clusters', transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_mapping/sampling_domain',
            self.sampling_map_callback, transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_mapping/navigation_goal_domain',
            self.navigation_goal_map_callback, transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_mapping/gmrf_domain',
            self.gmrf_map_callback, transient_qos())
        self.sampling_map = None
        self.navigation_goal_map = None
        self.gmrf_map = None
        self.planned = False

    def sampling_map_callback(self, msg):
        self.sampling_map = copy.deepcopy(msg)
        self.try_plan()

    def gmrf_map_callback(self, msg):
        self.gmrf_map = copy.deepcopy(msg)
        self.try_plan()

    def navigation_goal_map_callback(self, msg):
        self.navigation_goal_map = copy.deepcopy(msg)
        self.try_plan()

    def try_plan(self):
        if (self.planned or self.sampling_map is None or
                self.navigation_goal_map is None or self.gmrf_map is None):
            return
        selection, points, distance_oracle = self.build_points(
            self.sampling_map, self.navigation_goal_map, self.gmrf_map)
        self.publish_clusters(selection.partition)
        if selection.omitted_cluster_ids:
            self.get_logger().warning(
                'Navigation goal이 없는 LRS 클러스터를 제외합니다: '
                f'{list(selection.omitted_cluster_ids)}')
        if len(points) < 3:
            raise RuntimeError(
                'LRS requires at least 3 safe cluster representatives, '
                f'found {len(points)} from k={self.lrs_cluster_count}')

        distance_matrix = distance_oracle.matrix(points)
        start_distances = [
            distance_oracle.distance(
                (self.robot_start_x, self.robot_start_y),
                (point.x, point.y))
            for point in points]
        result = solve_lrs_tsp(
            points, (self.robot_start_x, self.robot_start_y),
            self.tsp_time_limit, distance_matrix=distance_matrix,
            start_distances=start_distances)
        ordered = [points[index] for index in result.tour]
        self.publish_route(ordered)
        self.publish_points(ordered)
        self.planned = True
        self.get_logger().info(
            f'LRS P1 complete: k={self.lrs_cluster_count}, '
            f'cluster_cells={len(selection.partition.cells)}, '
            f'points={len(points)}, '
            f'omitted={list(selection.omitted_cluster_ids)}, '
            f'inertia={selection.partition.inertia:.3f}, '
            f'status={result.status}, '
            f'objective={result.objective:.3f}m, gap={result.gap:.6f}, '
            f'cuts={result.cuts}')

    @staticmethod
    def domain_mask(domain_map, description):
        geometry = GridGeometry.from_message(domain_map)
        values = list(domain_map.data)
        expected = geometry.width * geometry.height
        if len(values) != expected:
            raise ValueError(
                f'{description} data size does not match its geometry')
        return geometry, np.asarray(values, dtype=np.int16).reshape(
            geometry.height, geometry.width) == 0

    def build_points(self, sampling_map, navigation_goal_map, gmrf_map):
        sampling_geometry, traversal_mask = self.domain_mask(
            sampling_map, 'sampling domain')
        goal_geometry, goal_mask = self.domain_mask(
            navigation_goal_map, 'navigation goal domain')
        if goal_geometry != sampling_geometry:
            raise ValueError(
                'navigation goal and sampling domain geometries differ')
        if np.any(goal_mask & ~traversal_mask):
            raise ValueError(
                'navigation goal domain must be a sampling domain subset')
        gmrf_geometry = GridGeometry.from_message(gmrf_map)
        selection = build_lrs_representatives(
            gmrf_geometry, sampling_geometry, traversal_mask, goal_mask,
            self.lrs_cluster_count, self.lrs_cluster_random_seed)
        points = [LrsPoint(
            point.field_row, point.field_col, point.x, point.y)
            for point in selection.representatives]
        return selection, points, SamplingDistanceOracle(
            sampling_geometry, traversal_mask)

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
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.30
        marker.color.r = marker.color.g = marker.color.b = 0.55
        marker.color.a = 1.0
        marker.points = [Point(x=p.x, y=p.y, z=0.15) for p in ordered]
        self.points_pub.publish(marker)
        log_marker = copy.deepcopy(marker)
        log_marker.ns = 'gas_mapping_lrs_status_log'
        self.points_log_pub.publish(log_marker)

    def publish_clusters(self, partition):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_lrs_clusters'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = 0.9 * float(
            self.gmrf_map.info.resolution)
        marker.scale.z = 0.04
        marker.points = [
            Point(x=cell.x, y=cell.y, z=0.02)
            for cell in partition.cells]
        palette = []
        for cluster_id in range(len(partition.centroids)):
            red, green, blue = colorsys.hsv_to_rgb(
                (0.61803398875 * cluster_id) % 1.0, 0.72, 1.0)
            palette.append(ColorRGBA(
                r=red, g=green, b=blue, a=0.25))
        marker.colors = [
            palette[cell.cluster_id] for cell in partition.cells]
        self.clusters_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = LrsPathPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
