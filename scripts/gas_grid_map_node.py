#!/usr/bin/env python3

import math
from copy import deepcopy
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from grid_graph_planner import GridGraphPlanner
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class GridCell:
    row: int
    col: int
    center_x: float
    center_y: float
    occupancy: int = 0
    inside_map: bool = False
    sampling_enabled: bool = False
    gas_estimate: float = math.nan
    sample_count: int = 0
    neighbors: list[tuple[int, int]] = field(default_factory=list)


class GasGridMapNode(Node):
    """Create a coarse 2-D data grid aligned with the existing /map."""

    def __init__(self):
        super().__init__('gas_grid_map_node')

        self.declare_parameter('grid_resolution', 0.5)
        self.declare_parameter('grid_margin', 0.5)
        self.declare_parameter('dfs_start_x', 0.0)
        self.declare_parameter('dfs_start_y', -4.0)
        self.grid_resolution = float(
            self.get_parameter('grid_resolution').value
        )
        self.grid_margin = float(self.get_parameter('grid_margin').value)
        self.dfs_start_x = float(self.get_parameter('dfs_start_x').value)
        self.dfs_start_y = float(self.get_parameter('dfs_start_y').value)
        self.grid_data = []
        self.dfs_route = []

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.publisher = self.create_publisher(
            OccupancyGrid, '/gas_grid', map_qos
        )
        self.lines_publisher = self.create_publisher(
            Marker, '/gas_grid_lines', map_qos
        )
        self.sampling_points_publisher = self.create_publisher(
            Marker, '/gas_grid_sampling_points', map_qos
        )
        self.dfs_path_publisher = self.create_publisher(
            Path, '/gas_grid/dfs_path', map_qos
        )
        self.dfs_endpoints_publisher = self.create_publisher(
            MarkerArray, '/gas_grid/dfs_endpoints', map_qos
        )
        self.subscription = self.create_subscription(
            OccupancyGrid, '/map', self.create_grid, map_qos
        )

    def create_grid(self, source_map):
        source_width_m = source_map.info.width * source_map.info.resolution
        source_height_m = source_map.info.height * source_map.info.resolution
        map_grid_width = math.floor(source_width_m / self.grid_resolution)
        map_grid_height = math.floor(source_height_m / self.grid_resolution)
        margin_cells = math.ceil(self.grid_margin / self.grid_resolution)
        grid_width = map_grid_width + 2 * margin_cells
        grid_height = map_grid_height + 2 * margin_cells
        grid_origin_x = (
            source_map.info.origin.position.x
            - margin_cells * self.grid_resolution
        )
        grid_origin_y = (
            source_map.info.origin.position.y
            - margin_cells * self.grid_resolution
        )

        # 실제 계산에 사용할 2차원 셀 배열: grid_data[row][column]
        self.grid_data = [
            [
                GridCell(
                    row=row,
                    col=col,
                    center_x=(
                        grid_origin_x + (col + 0.5) * self.grid_resolution
                    ),
                    center_y=(
                        grid_origin_y + (row + 0.5) * self.grid_resolution
                    ),
                    inside_map=(
                        margin_cells <= col < margin_cells + map_grid_width
                        and margin_cells <= row < margin_cells + map_grid_height
                    ),
                )
                for col in range(grid_width)
            ]
            for row in range(grid_height)
        ]

        # 확장 셀을 포함해 원본 map과 겹치는 모든 영역을 검사한다.
        # coarse cell 내부에 장애물 픽셀이 하나라도 있으면 100으로 저장한다.
        for row in range(grid_height):
            cell_y_min = (row - margin_cells) * self.grid_resolution
            cell_y_max = cell_y_min + self.grid_resolution
            overlap_y_min = max(0.0, cell_y_min)
            overlap_y_max = min(source_height_m, cell_y_max)

            if overlap_y_min >= overlap_y_max:
                continue

            source_row_start = max(0, math.floor(
                overlap_y_min / source_map.info.resolution
            ))
            source_row_end = min(source_map.info.height, math.ceil(
                overlap_y_max / source_map.info.resolution
            ))

            for col in range(grid_width):
                cell_x_min = (col - margin_cells) * self.grid_resolution
                cell_x_max = cell_x_min + self.grid_resolution
                overlap_x_min = max(0.0, cell_x_min)
                overlap_x_max = min(source_width_m, cell_x_max)

                if overlap_x_min >= overlap_x_max:
                    continue

                source_col_start = max(0, math.floor(
                    overlap_x_min / source_map.info.resolution
                ))
                source_col_end = min(source_map.info.width, math.ceil(
                    overlap_x_max / source_map.info.resolution
                ))

                values = [
                    source_map.data[source_row * source_map.info.width
                                    + source_col]
                    for source_row in range(source_row_start, source_row_end)
                    for source_col in range(source_col_start, source_col_end)
                ]

                if any(value == 100 for value in values):
                    self.grid_data[row][col].occupancy = 100
                elif any(value == -1 for value in values):
                    self.grid_data[row][col].occupancy = -1
                else:
                    self.grid_data[row][col].occupancy = 0

        for row in self.grid_data:
            for cell in row:
                cell.sampling_enabled = (
                    cell.inside_map and cell.occupancy == 0
                )

        planner = GridGraphPlanner(self.grid_data)
        edge_count = planner.build_neighbors()
        sampling_cell_count = planner.sampling_cell_count()
        start_cell = planner.nearest_sampling_cell(
            self.dfs_start_x, self.dfs_start_y
        )
        self.dfs_route = (
            planner.create_dfs_route(start_cell) if start_cell else []
        )
        visited_cell_count = len(set(self.dfs_route))

        if visited_cell_count < sampling_cell_count:
            self.get_logger().warning(
                f'DFS graph is disconnected: visited={visited_cell_count}, '
                f'total={sampling_cell_count}'
            )

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = source_map.header.frame_id or 'map'
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = grid_width
        grid_msg.info.height = grid_height
        grid_msg.info.origin = deepcopy(source_map.info.origin)
        grid_msg.info.origin.position.x = grid_origin_x
        grid_msg.info.origin.position.y = grid_origin_y
        grid_msg.data = [
            cell.occupancy
            for row in self.grid_data
            for cell in row
        ]

        self.publisher.publish(grid_msg)
        self.publish_grid_lines(grid_msg)
        self.publish_sampling_points(grid_msg)
        self.publish_dfs_path(grid_msg)
        self.publish_dfs_endpoints(grid_msg)
        self.get_logger().info(
            f'Published /gas_grid: {grid_width} x {grid_height}, '
            f'resolution={self.grid_resolution:.2f} m, '
            f'margin={margin_cells * self.grid_resolution:.2f} m, '
            f'graph_edges={edge_count}, '
            f'visited_cells={visited_cell_count}/{sampling_cell_count}, '
            f'dfs_steps={len(self.dfs_route)}'
        )

    def publish_grid_lines(self, grid_msg):
        marker = Marker()
        marker.header = grid_msg.header
        marker.ns = 'gas_grid'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.9

        origin_x = grid_msg.info.origin.position.x
        origin_y = grid_msg.info.origin.position.y
        resolution = grid_msg.info.resolution
        width_m = grid_msg.info.width * resolution
        height_m = grid_msg.info.height * resolution
        z = 0.03

        for col in range(grid_msg.info.width + 1):
            x = origin_x + col * resolution
            self.append_dashed_line(
                marker, x, origin_y, x, origin_y + height_m, z
            )

        for row in range(grid_msg.info.height + 1):
            y = origin_y + row * resolution
            self.append_dashed_line(
                marker, origin_x, y, origin_x + width_m, y, z
            )

        self.lines_publisher.publish(marker)

    @staticmethod
    def append_dashed_line(marker, x1, y1, x2, y2, z):
        dash_length = 0.12
        gap_length = 0.08
        length = math.hypot(x2 - x1, y2 - y1)
        direction_x = (x2 - x1) / length
        direction_y = (y2 - y1) / length
        distance = 0.0

        while distance < length:
            dash_end = min(distance + dash_length, length)
            marker.points.extend([
                Point(
                    x=x1 + direction_x * distance,
                    y=y1 + direction_y * distance,
                    z=z,
                ),
                Point(
                    x=x1 + direction_x * dash_end,
                    y=y1 + direction_y * dash_end,
                    z=z,
                ),
            ])
            distance += dash_length + gap_length

    def publish_sampling_points(self, grid_msg):
        marker = Marker()
        marker.header = grid_msg.header
        marker.ns = 'gas_grid_sampling_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.color.r = 0.0
        marker.color.g = 0.3
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.points = [
            Point(x=cell.center_x, y=cell.center_y, z=0.06)
            for row in self.grid_data
            for cell in row
            if cell.sampling_enabled
        ]

        self.sampling_points_publisher.publish(marker)

    def publish_dfs_path(self, grid_msg: OccupancyGrid) -> None:
        path = Path()
        path.header = grid_msg.header

        for row, col in self.dfs_route:
            cell = self.grid_data[row][col]
            pose = PoseStamped()
            pose.header = grid_msg.header
            pose.pose.position.x = cell.center_x
            pose.pose.position.y = cell.center_y
            pose.pose.position.z = 0.08
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.dfs_path_publisher.publish(path)

    def publish_dfs_endpoints(self, grid_msg: OccupancyGrid) -> None:
        markers = MarkerArray()
        if not self.dfs_route:
            self.dfs_endpoints_publisher.publish(markers)
            return

        start_row, start_col = self.dfs_route[0]
        end_row, end_col = self.dfs_route[-1]
        start_cell = self.grid_data[start_row][start_col]
        end_cell = self.grid_data[end_row][end_col]

        markers.markers.append(self.create_endpoint_marker(
            grid_msg=grid_msg,
            marker_id=0,
            cell=start_cell,
            diameter=0.22,
            red=0.0,
            green=1.0,
            blue=0.0,
        ))
        markers.markers.append(self.create_endpoint_marker(
            grid_msg=grid_msg,
            marker_id=1,
            cell=end_cell,
            diameter=0.14,
            red=1.0,
            green=0.0,
            blue=0.0,
        ))
        self.dfs_endpoints_publisher.publish(markers)

    @staticmethod
    def create_endpoint_marker(
        grid_msg: OccupancyGrid,
        marker_id: int,
        cell: GridCell,
        diameter: float,
        red: float,
        green: float,
        blue: float,
    ) -> Marker:
        marker = Marker()
        marker.header = grid_msg.header
        marker.ns = 'gas_grid_dfs_endpoints'
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = cell.center_x
        marker.pose.position.y = cell.center_y
        marker.pose.position.z = 0.10
        marker.pose.orientation.w = 1.0
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = 0.02
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = GasGridMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
