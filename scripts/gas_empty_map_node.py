#!/usr/bin/env python3
import copy
import json
import math
import os
import random
import tempfile

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


SOURCE_PARAMETER_NAMES = (
    'source_enabled', 'source_x', 'source_y',
    'source_strength', 'source_sigma',
)
RANDOM_SOURCE_PARAMETER_NAMES = (
    'source_mode', 'source_random_seed',
    'source_random_sigma_min', 'source_random_sigma_max',
    'source_random_min_separation', 'source_random_lrs_spacing',
    'source_random_detection_threshold',
)
HOTSPOT_SOURCE_PARAMETER_NAMES = (
    'source_hotspot_centers', 'source_hotspot_weights',
    'source_hotspot_jitter_sigma',
)
SOURCE_CONFIGURATION_PARAMETER_NAMES = (
    RANDOM_SOURCE_PARAMETER_NAMES + HOTSPOT_SOURCE_PARAMETER_NAMES)
SOURCE_MODES = (
    'manual', 'random_after_peak', 'recurrent_hotspots_after_peak')
AUTOMATIC_SOURCE_MODES = SOURCE_MODES[1:]
RANDOM_SOURCE_MAX_ATTEMPTS = 1000


def validated_source_state(state, min_x, max_x, min_y, max_y):
    enabled = state['source_enabled']
    if not isinstance(enabled, bool):
        raise ValueError('source_enabled must be boolean')
    x, y = float(state['source_x']), float(state['source_y'])
    strength = float(state['source_strength'])
    sigma = float(state['source_sigma'])
    if not all(math.isfinite(value) for value in (x, y, strength, sigma)):
        raise ValueError('source position, strength, and sigma must be finite')
    if not (min_x <= x <= max_x and min_y <= y <= max_y):
        raise ValueError('source position must be inside the map')
    if not 0.0 <= strength <= 1.0:
        raise ValueError('source_strength must be in [0, 1]')
    if sigma <= 0.0:
        raise ValueError('source_sigma must be positive')
    return {
        'source_enabled': enabled,
        'source_x': x, 'source_y': y,
        'source_strength': strength, 'source_sigma': sigma,
    }


def validated_random_source_config(config):
    mode = str(config['source_mode'])
    seed = config['source_random_seed']
    if mode not in SOURCE_MODES:
        raise ValueError(
            f'source_mode must be one of {", ".join(SOURCE_MODES)}')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('source_random_seed must be an integer')
    sigma_min = float(config['source_random_sigma_min'])
    sigma_max = float(config['source_random_sigma_max'])
    min_separation = float(config['source_random_min_separation'])
    lrs_spacing = float(config['source_random_lrs_spacing'])
    detection_threshold = float(
        config['source_random_detection_threshold'])
    numeric_values = (
        sigma_min, sigma_max, min_separation, lrs_spacing,
        detection_threshold)
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError('random source numeric parameters must be finite')
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError(
            'source random sigma range must satisfy 0 < min <= max')
    if min_separation < 0.0:
        raise ValueError('source_random_min_separation must be non-negative')
    if lrs_spacing <= 0.0:
        raise ValueError('source_random_lrs_spacing must be positive')
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError(
            'source_random_detection_threshold must be in [0, 1]')
    return {
        'source_mode': mode,
        'source_random_seed': seed,
        'source_random_sigma_min': sigma_min,
        'source_random_sigma_max': sigma_max,
        'source_random_min_separation': min_separation,
        'source_random_lrs_spacing': lrs_spacing,
        'source_random_detection_threshold': detection_threshold,
    }


def validated_hotspot_source_config(
        config, min_x, max_x, min_y, max_y):
    raw_centers = config['source_hotspot_centers']
    raw_weights = config['source_hotspot_weights']
    if (isinstance(raw_centers, (str, bytes))
            or not hasattr(raw_centers, '__iter__')):
        raise ValueError(
            'source_hotspot_centers must be a flat numeric array')
    if (isinstance(raw_weights, (str, bytes))
            or not hasattr(raw_weights, '__iter__')):
        raise ValueError('source_hotspot_weights must be a numeric array')
    try:
        centers = [float(value) for value in raw_centers]
        weights = [float(value) for value in raw_weights]
        jitter_sigma = float(config['source_hotspot_jitter_sigma'])
    except (TypeError, ValueError) as error:
        raise ValueError(
            'hotspot centers, weights, and jitter sigma must be numeric') \
            from error
    if len(centers) < 2 or len(centers) % 2 != 0:
        raise ValueError(
            'source_hotspot_centers must contain one or more x,y pairs')
    hotspot_count = len(centers) // 2
    if len(weights) != hotspot_count:
        raise ValueError(
            'source_hotspot_weights must have one value per hotspot center')
    if not all(math.isfinite(value) for value in (
            centers + weights + [jitter_sigma])):
        raise ValueError('hotspot source numeric parameters must be finite')
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError(
            'source_hotspot_weights must be non-negative with a positive sum')
    if jitter_sigma < 0.0:
        raise ValueError('source_hotspot_jitter_sigma must be non-negative')
    for index in range(hotspot_count):
        x = centers[2 * index]
        y = centers[2 * index + 1]
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            raise ValueError(
                f'hotspot center {index} must be inside the map')
    return {
        'source_hotspot_centers': centers,
        'source_hotspot_weights': weights,
        'source_hotspot_jitter_sigma': jitter_sigma,
    }


def inclusive_axis(minimum, maximum, spacing):
    if maximum < minimum:
        raise ValueError('map maximum must not be less than map minimum')
    count = int(math.floor((maximum - minimum) / spacing + 1e-9))
    return [minimum + index * spacing for index in range(count + 1)]


def random_source_is_lrs_detectable(
        state, min_x, max_x, min_y, max_y, lrs_spacing,
        detection_threshold):
    for y in inclusive_axis(min_y, max_y, lrs_spacing):
        for x in inclusive_axis(min_x, max_x, lrs_spacing):
            distance_sq = (
                (x - state['source_x']) ** 2
                + (y - state['source_y']) ** 2)
            concentration = math.exp(
                -distance_sq / (2.0 * state['source_sigma'] ** 2))
            if concentration >= detection_threshold:
                return True
    return False


def generate_random_source_state(
        rng, min_x, max_x, min_y, max_y, gmrf_resolution,
        sigma_min, sigma_max, min_separation, lrs_spacing,
        detection_threshold, previous_position=None,
        max_attempts=RANDOM_SOURCE_MAX_ATTEMPTS):
    x_centers = inclusive_axis(min_x, max_x, gmrf_resolution)
    y_centers = inclusive_axis(min_y, max_y, gmrf_resolution)
    min_separation_sq = min_separation ** 2
    for _ in range(max_attempts):
        proposed = {
            'source_enabled': True,
            'source_x': float(rng.choice(x_centers)),
            'source_y': float(rng.choice(y_centers)),
            'source_strength': 1.0,
            'source_sigma': float(rng.uniform(sigma_min, sigma_max)),
        }
        if previous_position is not None:
            dx = proposed['source_x'] - previous_position[0]
            dy = proposed['source_y'] - previous_position[1]
            if dx * dx + dy * dy < min_separation_sq:
                continue
        if random_source_is_lrs_detectable(
                proposed, min_x, max_x, min_y, max_y, lrs_spacing,
                detection_threshold):
            return proposed
    raise ValueError(
        f'could not generate a detectable random gas source in '
        f'{max_attempts} attempts')


def snap_to_grid(value, minimum, resolution):
    if not math.isfinite(float(value)):
        raise ValueError('grid coordinate must be finite')
    if not math.isfinite(float(resolution)) or resolution <= 0.0:
        raise ValueError('grid resolution must be positive and finite')
    grid_index = int(math.floor(
        (float(value) - minimum) / resolution + 0.5))
    return float(minimum + grid_index * resolution)


def generate_recurrent_hotspot_source_state(
        rng, min_x, max_x, min_y, max_y, gmrf_resolution,
        sigma_min, sigma_max, lrs_spacing, detection_threshold,
        hotspot_centers, hotspot_weights, hotspot_jitter_sigma,
        max_attempts=RANDOM_SOURCE_MAX_ATTEMPTS):
    hotspots = [
        (hotspot_centers[index], hotspot_centers[index + 1])
        for index in range(0, len(hotspot_centers), 2)]
    for _ in range(max_attempts):
        hotspot_index = rng.choices(
            range(len(hotspots)), weights=hotspot_weights, k=1)[0]
        center_x, center_y = hotspots[hotspot_index]
        proposed = {
            'source_enabled': True,
            'source_x': snap_to_grid(
                rng.gauss(center_x, hotspot_jitter_sigma),
                min_x, gmrf_resolution),
            'source_y': snap_to_grid(
                rng.gauss(center_y, hotspot_jitter_sigma),
                min_y, gmrf_resolution),
            'source_strength': 1.0,
            'source_sigma': float(rng.uniform(sigma_min, sigma_max)),
        }
        if not (
                min_x <= proposed['source_x'] <= max_x
                and min_y <= proposed['source_y'] <= max_y):
            continue
        if random_source_is_lrs_detectable(
                proposed, min_x, max_x, min_y, max_y, lrs_spacing,
                detection_threshold):
            return proposed
    raise ValueError(
        f'could not generate a detectable recurrent-hotspot gas source in '
        f'{max_attempts} attempts')


def load_source_state(path):
    expanded = os.path.expanduser(str(path))
    if not os.path.exists(expanded):
        return None
    with open(expanded, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if int(payload.get('version', -1)) != 1:
        raise ValueError('unsupported gas source state version')
    source = payload.get('source')
    if not isinstance(source, dict) or any(
            name not in source for name in SOURCE_PARAMETER_NAMES):
        raise ValueError('gas source state is incomplete')
    return {name: source[name] for name in SOURCE_PARAMETER_NAMES}


def save_source_state(path, state):
    expanded = os.path.expanduser(str(path))
    directory = os.path.dirname(expanded) or '.'
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', dir=directory, prefix='.gas_source_state_',
                suffix='.json', delete=False, encoding='utf-8') as stream:
            json.dump({'version': 1, 'source': state}, stream, indent=2)
            temporary_path = stream.name
        os.replace(temporary_path, expanded)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class EmptyGasMapNode(Node):
    def __init__(self):
        super().__init__('gas_empty_map_node')
        defaults = {
            'source_enabled': False,
            'source_x': 10.0, 'source_y': 10.0, 'source_strength': 1.0,
            'source_sigma': 5.0,
            'source_mode': 'manual',
            'source_random_seed': -1,
            'source_random_sigma_min': 4.0,
            'source_random_sigma_max': 7.0,
            'source_random_min_separation': 10.0,
            'source_random_lrs_spacing': 10.0,
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
            'nav_resolution': 0.05, 'ground_truth_resolution': 0.1,
            'gmrf_resolution': 1.0, 'map_margin': 2.5,
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
            self.nav_resolution, self.ground_truth_resolution,
            self.gmrf_resolution, self.map_margin,
            self.log_concentration_min)
        if not all(math.isfinite(float(value))
                   for value in map_config_values):
            raise ValueError('map and grid numeric parameters must be finite')
        if min(self.nav_resolution, self.ground_truth_resolution,
               self.gmrf_resolution) <= 0.0:
            raise ValueError('grid resolutions must be positive')
        if self.map_margin < 0.0:
            raise ValueError('map_margin must be non-negative')
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
        if self.source_mode == 'manual':
            self.restore_source_state()
        else:
            self.initialize_random_source()
        configured_source = validated_source_state(
            {name: getattr(self, name) for name in SOURCE_PARAMETER_NAMES},
            self.map_min_x, self.map_max_x,
            self.map_min_y, self.map_max_y)
        for name, value in configured_source.items():
            setattr(self, name, value)

        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map', transient_qos())
        self.gt_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/ground_truth', transient_qos())
        self.gt_log_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/ground_truth_log', transient_qos())
        self.gmrf_grid_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/gmrf_domain', transient_qos())
        self.source_pub = self.create_publisher(
            Marker, '/gas_mapping/source', transient_qos())
        self.sensor_pub = self.create_publisher(
            Float64, '/gas_mapping/sensor_concentration', 20)
        self.create_subscription(PoseStamped, '/gas_sensor/sensor_pose', self.pose_callback, 20)
        self.latest_sensor_pose = None
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.source_advance_service = self.create_service(
            Trigger, '/gas_mapping/source/advance',
            self.advance_source_callback)
        self.initialize_maps()

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
        result = self.set_parameters_atomically([
            Parameter(name=name, value=proposed[name])
            for name in SOURCE_PARAMETER_NAMES])
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
                self.source_random_lrs_spacing,
                self.source_random_detection_threshold,
                self.source_hotspot_centers,
                self.source_hotspot_weights,
                self.source_hotspot_jitter_sigma)
        return generate_random_source_state(
            self.source_rng,
            self.map_min_x, self.map_max_x,
            self.map_min_y, self.map_max_y,
            self.gmrf_resolution,
            self.source_random_sigma_min,
            self.source_random_sigma_max,
            self.source_random_min_separation,
            self.source_random_lrs_spacing,
            self.source_random_detection_threshold,
            previous_position=previous_position)

    def advance_source_callback(self, request, response):
        del request
        if self.source_mode == 'manual':
            response.success = True
            response.message = 'changed=false; manual source mode'
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
    node = EmptyGasMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
