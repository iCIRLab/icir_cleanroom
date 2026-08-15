#!/usr/bin/env python3
import copy
import math

import numpy as np
import rclpy
from scipy.ndimage import map_coordinates
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Float32MultiArray, Float64
from visualization_msgs.msg import Marker

from gmrf_grid import GmrfGrid


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class GasMappingControllerNode(Node):
    def __init__(self):
        super().__init__('gas_mapping_controller_node')
        params = {
            'dwell_seconds': 1.0, 'max_retries': 2,
            'observation_variance_scale': 0.01,
            'observation_variance_floor': 0.001,
            'smoothness_precision': 1.0,
            'cardinal_weight': 0.75, 'diagonal_weight': 0.25,
            'background_precision': 1.0, 'background_mean': 0.0,
            'estimate_resolution': 0.1, 'log_concentration_min': -4.0,
        }
        for name, default in params.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)

        self.create_subscription(Path, '/gas_lrs/tsp_route', self.path_callback, transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_lrs/gmrf_domain', self.map_callback, transient_qos())
        self.create_subscription(PoseStamped, '/gas_sensor/sensor_pose', self.pose_callback, 30)
        self.create_subscription(Float64, '/gas_lrs/sensor_concentration', self.value_callback, 30)
        self.estimate_pub = self.create_publisher(
            OccupancyGrid, '/gas_lrs/estimate', transient_qos())
        self.estimate_log_pub = self.create_publisher(
            OccupancyGrid, '/gas_lrs/estimate_log', transient_qos())
        self.poses_pub = self.create_publisher(
            PoseArray, '/gas_lrs/measurements/poses', transient_qos())
        self.values_pub = self.create_publisher(
            Float32MultiArray, '/gas_lrs/measurements/values', transient_qos())
        self.status_pub = self.create_publisher(
            Marker, '/gas_lrs/sampling_status', transient_qos())
        self.status_log_pub = self.create_publisher(
            Marker, '/gas_lrs/sampling_status_log', transient_qos())
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.map_msg = None
        self.gmrf = None
        self.goals = []
        self.current = 0
        self.retry = 0
        self.started = False
        self.returning = False
        self.latest_pose = None
        self.measurement_pose = None
        self.dwell_values = []
        self.measuring = False
        self.dwell_timer = None
        self.measured_poses = []
        self.measured_values = []
        self.status_values = []
        self.start_timer = self.create_timer(1.0, self.try_start)
        self.get_logger().info('LRS route, GMRF domain, sensor pose와 Nav2를 기다리는 중...')

    def path_callback(self, msg):
        if not self.started and len(msg.poses) >= 4:
            self.goals = list(msg.poses[:-1])
            self.status_values = [None] * len(self.goals)
            self.get_logger().info(f'LRS closed route 수신: {len(self.goals)} points')

    def map_callback(self, msg):
        if self.gmrf is not None:
            return
        self.map_msg = copy.deepcopy(msg)
        self.gmrf = GmrfGrid(
            msg, self.smoothness_precision, self.background_precision,
            self.background_mean, self.observation_variance_scale,
            self.observation_variance_floor, self.cardinal_weight,
            self.diagonal_weight)
        self.publish_estimate()
        self.get_logger().info(f'GMRF 생성: {len(self.gmrf.solution)} free variables')

    def pose_callback(self, msg):
        self.latest_pose = copy.deepcopy(msg)

    def value_callback(self, msg):
        if self.measuring:
            self.dwell_values.append(float(msg.data))

    def try_start(self):
        if self.started or not self.goals or self.gmrf is None or self.latest_pose is None:
            return
        if not self.nav_client.server_is_ready():
            return
        self.started = True
        self.start_timer.cancel()
        self.publish_status()
        self.send_goal()

    def send_goal(self):
        if self.current >= len(self.goals):
            if self.returning:
                self.get_logger().info('=== LRS 1회 순찰 및 시작점 복귀 완료 ===')
                return
            self.returning = True
            target = self.goals[0]
            label = '시작 LRS 포인트 복귀'
        else:
            target = self.goals[self.current]
            label = f'LRS [{self.current + 1}/{len(self.goals)}]'
        goal = NavigateToPose.Goal()
        goal.pose = copy.deepcopy(target)
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        next_index = (self.current + 1) % len(self.goals)
        next_pose = self.goals[next_index].pose.position
        yaw = math.atan2(next_pose.y - target.pose.position.y,
                         next_pose.x - target.pose.position.x)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.get_logger().info(
            f'{label} -> ({target.pose.position.x:.2f}, {target.pose.position.y:.2f})')
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.navigation_failed('goal rejected')
            return
        result = handle.get_result_async()
        result.add_done_callback(self.goal_result)

    def goal_result(self, future):
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.navigation_failed(f'status={status}')
            return
        self.retry = 0
        if self.returning:
            self.current = len(self.goals)
            self.send_goal()
            return
        self.measurement_pose = copy.deepcopy(self.latest_pose)
        self.dwell_values = []
        self.measuring = True
        self.dwell_timer = self.create_timer(float(self.dwell_seconds), self.finish_dwell)

    def finish_dwell(self):
        self.dwell_timer.cancel()
        self.dwell_timer = None
        self.measuring = False
        if not self.dwell_values:
            self.get_logger().warning('농도 표본이 없어 이 포인트를 미측정으로 남깁니다')
        else:
            value = sum(self.dwell_values) / len(self.dwell_values)
            pose = self.measurement_pose.pose
            self.measured_poses.append(copy.deepcopy(pose))
            self.measured_values.append(value)
            self.status_values[self.current] = value
            cell = self.gmrf.add_observation(pose.position.x, pose.position.y, value)
            converged, info = self.gmrf.solve()
            if converged:
                self.publish_estimate()
            else:
                self.get_logger().warning(
                    f'GMRF CG 미수렴(info={info}); 마지막 정상 지도를 유지합니다')
            self.publish_measurements()
            self.get_logger().info(
                f'측정 완료: pose=({pose.position.x:.3f},{pose.position.y:.3f}), '
                f'mean={value:.4f}, samples={len(self.dwell_values)}, cell={cell}')
        self.current += 1
        self.publish_status()
        self.send_goal()

    def navigation_failed(self, reason):
        self.retry += 1
        if self.retry <= int(self.max_retries):
            self.get_logger().warning(f'이동 실패({reason}), 재시도 {self.retry}/{self.max_retries}')
        else:
            self.get_logger().warning(f'이동 실패({reason}), 미측정으로 남기고 다음 포인트 진행')
            self.retry = 0
            if self.returning:
                return
            self.current += 1
            self.publish_status()
        self.send_goal()

    def publish_estimate(self):
        msg = copy.deepcopy(self.map_msg)
        msg.header.stamp = self.get_clock().now().to_msg()
        model_resolution = self.gmrf.resolution
        display_resolution = float(self.estimate_resolution)
        if display_resolution <= 0.0:
            raise ValueError('estimate_resolution must be positive')
        if float(self.log_concentration_min) >= 0.0:
            raise ValueError('log_concentration_min must be negative')
        if math.isclose(display_resolution, model_resolution):
            field = np.zeros((self.gmrf.height, self.gmrf.width), dtype=float)
            field[self.gmrf.free] = self.gmrf.solution
            self.publish_estimate_grids(msg, field, self.gmrf.free)
            return

        # The GMRF variables lie at cell centers. Publish only the 0..50 m
        # model-center domain so the estimate aligns exactly with the GT map.
        min_x = self.gmrf.origin_x + 0.5 * model_resolution
        min_y = self.gmrf.origin_y + 0.5 * model_resolution
        max_x = self.gmrf.origin_x + (self.gmrf.width - 0.5) * model_resolution
        max_y = self.gmrf.origin_y + (self.gmrf.height - 0.5) * model_resolution
        width = int(round((max_x - min_x) / display_resolution)) + 1
        height = int(round((max_y - min_y) / display_resolution)) + 1

        field = np.zeros((self.gmrf.height, self.gmrf.width), dtype=float)
        field[self.gmrf.free] = self.gmrf.solution
        x_world = min_x + np.arange(width) * display_resolution
        y_world = min_y + np.arange(height) * display_resolution
        x_index = (x_world - min_x) / model_resolution
        y_index = (y_world - min_y) / model_resolution
        grid_x, grid_y = np.meshgrid(x_index, y_index)
        interpolated = map_coordinates(
            field, [grid_y, grid_x], order=1, mode='nearest')

        msg.info.resolution = display_resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = min_x - 0.5 * display_resolution
        msg.info.origin.position.y = min_y - 0.5 * display_resolution
        display_free = np.ones((height, width), dtype=bool)
        self.publish_estimate_grids(msg, interpolated, display_free)

    def publish_estimate_grids(self, template, values, free_mask):
        linear = copy.deepcopy(template)
        linear_data = np.full(values.shape, -1, dtype=np.int16)
        clipped = np.clip(values, 0.0, 1.0)
        linear_data[free_mask] = np.rint(100.0 * clipped[free_mask]).astype(
            np.int16)
        linear.data = linear_data.astype(np.int8).ravel().tolist()
        self.estimate_pub.publish(linear)

        # Fig. 3 of the LHR-GDM paper displays log concentration over [-4, 0].
        # Keep the linear topic above for data consumers and publish this second
        # grid only as a paper-comparison visualization.
        log_min = float(self.log_concentration_min)
        concentration_floor = math.exp(log_min)
        log_values = np.log(np.clip(clipped, concentration_floor, 1.0))
        log_normalized = np.clip((log_values - log_min) / -log_min, 0.0, 1.0)
        log_data = np.full(values.shape, -1, dtype=np.int16)
        log_data[free_mask] = np.rint(
            100.0 * log_normalized[free_mask]).astype(np.int16)
        log_grid = copy.deepcopy(template)
        log_grid.data = log_data.astype(np.int8).ravel().tolist()
        self.estimate_log_pub.publish(log_grid)

    def publish_measurements(self):
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = self.get_clock().now().to_msg()
        poses.poses = list(self.measured_poses)
        self.poses_pub.publish(poses)
        self.values_pub.publish(Float32MultiArray(data=self.measured_values))

    def publish_status(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_lrs_sampling_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.18
        marker.points = [Point(x=g.pose.position.x, y=g.pose.position.y, z=0.08)
                         for g in self.goals]
        marker.colors = []
        for index in range(len(self.goals)):
            if self.status_values[index] is not None:
                value = self.status_values[index]
                marker.colors.append(ColorRGBA(r=value, g=0.0, b=1.0 - value, a=1.0))
            else:
                marker.colors.append(ColorRGBA(r=0.55, g=0.55, b=0.55, a=1.0))
        self.status_pub.publish(marker)

        log_marker = copy.deepcopy(marker)
        log_marker.ns = 'gas_lrs_sampling_points_log'
        log_marker.colors = []
        log_min = float(self.log_concentration_min)
        concentration_floor = math.exp(log_min)
        for value in self.status_values:
            if value is None:
                log_marker.colors.append(ColorRGBA(
                    r=0.55, g=0.55, b=0.55, a=1.0))
                continue
            clipped = min(1.0, max(concentration_floor, float(value)))
            display_value = min(1.0, max(
                0.0, (math.log(clipped) - log_min) / -log_min))
            log_marker.colors.append(ColorRGBA(
                r=display_value, g=0.0, b=1.0 - display_value, a=1.0))
        self.status_log_pub.publish(log_marker)


def main(args=None):
    rclpy.init(args=args)
    node = GasMappingControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
