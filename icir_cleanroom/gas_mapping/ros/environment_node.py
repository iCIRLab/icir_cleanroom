"""ROS adapter for the synthetic gas environment."""
import copy
import json
import math
import os
import random

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from ..environment import (
    AUTOMATIC_SOURCE_MODES, HOTSPOT_SOURCE_PARAMETER_NAMES,
    RANDOM_SOURCE_PARAMETER_NAMES, SOURCE_CONFIGURATION_PARAMETER_NAMES,
    SOURCE_PARAMETER_NAMES, generate_random_source_state,
    generate_recurrent_hotspot_source_state, load_source_state,
    save_source_state, validated_hotspot_source_config,
    validated_random_source_config, validated_source_state)
from ..mapping.domains import (
    field_geometry, navigation_goal_mask, sampling_mask)
from ..mapping.grid_geometry import GridGeometry
from ..mapping.lrs_representatives import build_lrs_representatives


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class GasEnvironmentNode(Node):
    def __init__(self):
        super().__init__('gas_environment_node')
        defaults = {
            'source_enabled': False,
            'source_x': 10.0, 'source_y': 10.0, 'source_strength': 1.0,
            'source_sigma': 5.0,
            'source_mode': 'manual',
            'source_random_seed': -1,
            'source_random_sigma_min': 4.0,
            'source_random_sigma_max': 7.0,
            'source_random_min_separation': 10.0,
            'lrs_cluster_count': 36,
            'lrs_cluster_random_seed': 0,
            'source_random_detection_threshold': 0.2,
            'source_hotspot_centers': [
                10.0, 10.0,
                40.0, 10.0,
                15.0, 40.0,
                40.0, 40.0,
            ],
            'source_hotspot_weights': [0.4, 0.3, 0.2, 0.1],
            'source_hotspot_jitter_sigma': 3.0,
            'map_min_x': 0.0, 'map_max_x': 50.0,
            'map_min_y': 0.0, 'map_max_y': 50.0,
            'ground_truth_resolution': 0.1, 'gmrf_resolution': 1.0,
            'sampling_clearance': 0.25,
            'navigation_goal_clearance': 0.25,
            'robot_start_x': 0.0, 'robot_start_y': 0.0,
            'log_concentration_min': -4.0,
            'persist_source_state': True,
            'source_state_file': '~/.ros/icir_cleanroom/gas_source_state.json',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        self._internal_source_update = False
        map_config_values = (
            self.map_min_x, self.map_max_x,
            self.map_min_y, self.map_max_y,
            self.ground_truth_resolution, self.gmrf_resolution,
            self.sampling_clearance, self.navigation_goal_clearance,
            self.robot_start_x, self.robot_start_y,
            self.log_concentration_min)
        if not all(math.isfinite(float(value))
                   for value in map_config_values):
            raise ValueError('map and grid numeric parameters must be finite')
        if min(self.ground_truth_resolution, self.gmrf_resolution) <= 0.0:
            raise ValueError('grid resolutions must be positive')
        if (isinstance(self.lrs_cluster_count, bool) or
                not isinstance(self.lrs_cluster_count, int) or
                self.lrs_cluster_count <= 0):
            raise ValueError(
                'lrs_cluster_count must be a positive integer')
        if (isinstance(self.lrs_cluster_random_seed, bool) or
                not isinstance(self.lrs_cluster_random_seed, int)):
            raise ValueError(
                'lrs_cluster_random_seed must be an integer')
        if self.sampling_clearance < 0.0:
            raise ValueError('sampling_clearance must be non-negative')
        if self.navigation_goal_clearance <= 0.0:
            raise ValueError(
                'navigation_goal_clearance must be positive')
        if (self.map_max_x < self.map_min_x
                or self.map_max_y < self.map_min_y):
            raise ValueError('map maximums must not be less than minimums')
        if self.log_concentration_min >= 0.0:
            raise ValueError('log_concentration_min must be negative')
        random_config = validated_random_source_config({
            name: getattr(self, name)
            for name in RANDOM_SOURCE_PARAMETER_NAMES})
        for name, value in random_config.items():
            setattr(self, name, value)
        hotspot_config = validated_hotspot_source_config(
            {name: getattr(self, name)
             for name in HOTSPOT_SOURCE_PARAMETER_NAMES},
            self.map_min_x, self.map_max_x,
            self.map_min_y, self.map_max_y)
        for name, value in hotspot_config.items():
            setattr(self, name, value)
        self.source_rng = random.Random(
            None if self.source_random_seed < 0
            else self.source_random_seed)
        self.source_detection_points = None
        self.source_ready = self.source_mode == 'manual'
        if self.source_mode == 'manual':
            self.restore_source_state()
        configured_source = validated_source_state(
            {name: getattr(self, name) for name in SOURCE_PARAMETER_NAMES},
            self.map_min_x, self.map_max_x,
            self.map_min_y, self.map_max_y)
        for name, value in configured_source.items():
            setattr(self, name, value)

        self.gt_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/ground_truth', transient_qos())
        self.gt_log_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/ground_truth_log', transient_qos())
        self.gmrf_grid_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/gmrf_domain', transient_qos())
        self.sampling_grid_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/sampling_domain', transient_qos())
        self.navigation_goal_grid_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/navigation_goal_domain',
            transient_qos())
        self.source_pub = self.create_publisher(
            Marker, '/gas_mapping/source', transient_qos())
        self.sensor_pub = self.create_publisher(
            Float64, '/gas_mapping/sensor_concentration', 20)
        self.create_subscription(PoseStamped, '/gas_sensor/sensor_pose', self.pose_callback, 20)
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, transient_qos())
        self.latest_sensor_pose = None
        self.map_initialized = False
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.source_advance_service = self.create_service(
            Trigger, '/gas_mapping/source/advance',
            self.advance_source_callback)

    def parameter_callback(self, parameters):
        startup_only_changes = [
            parameter.name for parameter in parameters
            if parameter.name in SOURCE_CONFIGURATION_PARAMETER_NAMES]
        if startup_only_changes:
            return SetParametersResult(
                successful=False,
                reason=(
                    'automatic source configuration is startup-only; '
                    'restart the node to change it'))
        proposed = {
            name: getattr(self, name) for name in SOURCE_PARAMETER_NAMES}
        changed = False
        for parameter in parameters:
            if parameter.name in SOURCE_PARAMETER_NAMES:
                proposed[parameter.name] = parameter.value
                changed = True
        if not changed:
            return SetParametersResult(successful=True)
        if (self.source_mode in AUTOMATIC_SOURCE_MODES
                and not self._internal_source_update):
            return SetParametersResult(
                successful=False,
                reason=(
                    f'source parameters are managed by {self.source_mode} '
                    'mode'))
        try:
            proposed = validated_source_state(
                proposed, self.map_min_x, self.map_max_x,
                self.map_min_y, self.map_max_y)
            if self.source_mode in AUTOMATIC_SOURCE_MODES:
                if (not proposed['source_enabled']
                        or proposed['source_strength'] != 1.0):
                    raise ValueError(
                        'random source must be enabled with strength 1.0')
                if not (
                        self.source_random_sigma_min
                        <= proposed['source_sigma']
                        <= self.source_random_sigma_max):
                    raise ValueError(
                        'random source sigma is outside the configured range')
            if self.persist_source_state:
                save_source_state(self.source_state_file, proposed)
        except (OSError, TypeError, ValueError) as error:
            return SetParametersResult(
                successful=False, reason=str(error))
        for name in SOURCE_PARAMETER_NAMES:
            setattr(self, name, proposed[name])
        if self.map_initialized:
            self.publish_ground_truth()
            self.publish_source()
        if self.latest_sensor_pose is not None:
            self.publish_sensor_value(self.latest_sensor_pose)
        self.get_logger().info(
            'Gas source updated: '
            f'enabled={self.source_enabled}, '
            f'position=({self.source_x:.3f},{self.source_y:.3f}), '
            f'strength={self.source_strength:.3f}, '
            f'sigma={self.source_sigma:.3f}')
        return SetParametersResult(successful=True)

    def initialize_random_source(self):
        proposed = self.generate_random_source(previous_position=None)
        self._internal_source_update = True
        try:
            result = self.set_parameters_atomically([
                Parameter(name=name, value=proposed[name])
                for name in SOURCE_PARAMETER_NAMES])
        finally:
            self._internal_source_update = False
        if not result.successful:
            raise ValueError(
                f'failed to install initial random source: {result.reason}')
        if self.persist_source_state:
            save_source_state(self.source_state_file, proposed)
        for name in SOURCE_PARAMETER_NAMES:
            setattr(self, name, proposed[name])
        self.get_logger().info(
            'Initial random gas source generated: '
            f'position=({self.source_x:.3f},{self.source_y:.3f}), '
            f'strength={self.source_strength:.3f}, '
            f'sigma={self.source_sigma:.3f}')

    def generate_random_source(self, previous_position):
        if self.source_mode == 'recurrent_hotspots_after_peak':
            return generate_recurrent_hotspot_source_state(
                self.source_rng,
                self.map_min_x, self.map_max_x,
                self.map_min_y, self.map_max_y,
                self.gmrf_resolution,
                self.source_random_sigma_min,
                self.source_random_sigma_max,
                self.source_random_detection_threshold,
                self.source_hotspot_centers,
                self.source_hotspot_weights,
                self.source_hotspot_jitter_sigma,
                self.source_detection_points)
        return generate_random_source_state(
            self.source_rng,
            self.map_min_x, self.map_max_x,
            self.map_min_y, self.map_max_y,
            self.gmrf_resolution,
            self.source_random_sigma_min,
            self.source_random_sigma_max,
            self.source_random_min_separation,
            self.source_random_detection_threshold,
            self.source_detection_points,
            previous_position=previous_position,
        )

    def advance_source_callback(self, request, response):
        del request
        if self.source_mode == 'manual':
            response.success = True
            response.message = 'changed=false; manual source mode'
            return response
        if not self.map_initialized:
            response.success = False
            response.message = 'changed=false; sampling domain is not ready'
            return response
        try:
            proposed = self.generate_random_source(
                previous_position=(self.source_x, self.source_y))
            self._internal_source_update = True
            try:
                result = self.set_parameters_atomically([
                    Parameter(name=name, value=proposed[name])
                    for name in SOURCE_PARAMETER_NAMES])
            finally:
                self._internal_source_update = False
            if not result.successful:
                raise ValueError(result.reason or 'source update was rejected')
        except (OSError, TypeError, ValueError) as error:
            response.success = False
            response.message = f'Random source advance failed: {error}'
            self.get_logger().error(response.message)
            return response
        response.success = True
        response.message = 'changed=true; automatic source advanced'
        self.get_logger().info(
            'Random gas source advanced: '
            f'position=({self.source_x:.3f},{self.source_y:.3f}), '
            f'strength={self.source_strength:.3f}, '
            f'sigma={self.source_sigma:.3f}')
        return response

    def restore_source_state(self):
        if not self.persist_source_state:
            return
        try:
            restored = load_source_state(self.source_state_file)
            if restored is None:
                return
            restored = validated_source_state(
                restored, self.map_min_x, self.map_max_x,
                self.map_min_y, self.map_max_y)
            results = self.set_parameters([
                Parameter(name=name, value=restored[name])
                for name in SOURCE_PARAMETER_NAMES])
            if not all(result.successful for result in results):
                raise ValueError('failed to restore ROS source parameters')
            for name in SOURCE_PARAMETER_NAMES:
                setattr(self, name, restored[name])
            self.get_logger().info(
                f'Persisted gas source state restored: '
                f'{os.path.expanduser(str(self.source_state_file))}')
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.get_logger().warning(
                f'Gas source state could not be restored; using launch '
                f'defaults: {error}')

    def concentration(self, x, y):
        if not self.source_enabled:
            return 0.0
        distance_sq = (x - self.source_x) ** 2 + (y - self.source_y) ** 2
        return min(1.0, max(0.0, self.source_strength * math.exp(
            -distance_sq / (2.0 * self.source_sigma ** 2))))

    def pose_callback(self, msg):
        self.latest_sensor_pose = copy.deepcopy(msg)
        self.publish_sensor_value(msg)

    def publish_sensor_value(self, msg):
        self.sensor_pub.publish(Float64(data=self.concentration(
            msg.pose.position.x, msg.pose.position.y)))

    def make_grid(self, resolution, min_x, min_y, max_x, max_y):
        geometry = field_geometry(
            min_x, max_x, min_y, max_y, resolution)
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(resolution)
        msg.info.width = geometry.width
        msg.info.height = geometry.height
        msg.info.origin.position.x = geometry.origin_x
        msg.info.origin.position.y = geometry.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = [0] * (msg.info.width * msg.info.height)
        return msg

    def make_domain_grid(self, nav_map, mask):
        output = copy.deepcopy(nav_map)
        output.header.stamp = self.get_clock().now().to_msg()
        navigation_values = list(nav_map.data)
        output.data = [
            0 if is_accessible else (
                -1 if navigation_values[index] < 0 else 100)
            for index, is_accessible in enumerate(mask.ravel())]
        return output

    def map_callback(self, nav_map):
        if self.map_initialized:
            return
        try:
            accessible = sampling_mask(
                nav_map, self.sampling_clearance,
                self.robot_start_x, self.robot_start_y)
            navigation_goals = navigation_goal_mask(
                nav_map, accessible, self.navigation_goal_clearance)
        except ValueError as error:
            self.get_logger().error(f'Navigation domain 생성 실패: {error}')
            return
        sampling_map = self.make_domain_grid(nav_map, accessible)
        navigation_goal_map = self.make_domain_grid(
            nav_map, navigation_goals)
        sampling_geometry = GridGeometry.from_message(nav_map)
        gmrf_map = self.make_grid(
            self.gmrf_resolution,
            self.map_min_x, self.map_min_y,
            self.map_max_x, self.map_max_y)
        selection = build_lrs_representatives(
            GridGeometry.from_message(gmrf_map), sampling_geometry,
            accessible, navigation_goals,
            self.lrs_cluster_count, self.lrs_cluster_random_seed)
        self.source_detection_points = [
            (point.x, point.y) for point in selection.representatives]
        if selection.omitted_cluster_ids:
            self.get_logger().warning(
                'Navigation goal이 없는 source-detection 클러스터를 '
                f'제외합니다: {list(selection.omitted_cluster_ids)}')
        if len(self.source_detection_points) < 3:
            raise RuntimeError(
                'Source detection requires at least 3 safe cluster '
                f'representatives, found {len(self.source_detection_points)} '
                f'from k={self.lrs_cluster_count}')
        if not self.source_ready:
            self.initialize_random_source()
            self.source_ready = True
        self.map_initialized = True
        self.gmrf_grid_pub.publish(gmrf_map)
        self.sampling_grid_pub.publish(sampling_map)
        self.navigation_goal_grid_pub.publish(navigation_goal_map)
        self.publish_ground_truth()
        self.publish_source()
        self.get_logger().info(
            f'Gas environment ready: nav={nav_map.info.resolution:.3f}m, '
            f'GT={self.ground_truth_resolution:.3f}m, '
            f'GMRF={self.gmrf_resolution:.3f}m, '
            f'sampling_cells={int(accessible.sum())}, '
            f'navigation_goal_cells={int(navigation_goals.sum())}, '
            f'navigation_goal_clearance='
            f'{float(self.navigation_goal_clearance):.3f}m')

    def publish_ground_truth(self):
        output = self.make_grid(
            self.ground_truth_resolution,
            self.map_min_x, self.map_min_y,
            self.map_max_x, self.map_max_y)
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
        marker.ns = 'gas_mapping_source'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD if self.source_enabled else Marker.DELETE
        marker.pose.position.x = self.source_x
        marker.pose.position.y = self.source_y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.4
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.1, 0.0, 1.0
        self.source_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GasEnvironmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
