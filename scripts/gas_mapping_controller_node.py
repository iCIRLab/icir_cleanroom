#!/usr/bin/env python3
"""Continuously patrol with LRS, adaptive HRS, and peak confirmation."""

import copy
from concurrent.futures import ThreadPoolExecutor
import math
import time

import numpy as np
import rclpy
from scipy.ndimage import map_coordinates
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, ColorRGBA, Float32MultiArray, Float64, Int32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from gas_history import GasHistoryStore, history_adjusted_cycle
from gmrf_grid import (
    GmrfGrid, higher_measured_neighbor, normalized_ucb, peak_search_status)
from hrs_path_planner import (
    HrsCell, PLANNER_MODES, solve_hrs_p2, solve_peak_confirmation_route)
from lrs_priority_planner import (
    LrsRewardPoint, lrs_coverage_fallback, solve_lrs_priority_route)


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class GasMappingControllerNode(Node):
    def __init__(self):
        super().__init__('gas_mapping_controller_node')
        params = {
            'lrs_dwell_seconds': 1.0, 'hrs_dwell_seconds': 2.0,
            'max_retries': 2, 'max_cell_failures': 3,
            'observation_variance_scale': 0.01,
            'observation_variance_floor': 0.001,
            'smoothness_precision': 1.0,
            'cardinal_weight': 0.75, 'diagonal_weight': 0.25,
            'background_precision': 1.0, 'background_mean': 0.0,
            'estimate_resolution': 0.1, 'log_concentration_min': -4.0,
            'gabp_max_iterations': 500, 'gabp_tolerance': 1.0e-6,
            'gabp_damping': 0.5, 'gabp_retry_damping': 0.25,
            'gabp_cg_warning_tolerance': 1.0e-4,
            'hrs_ucb_k': 1.0, 'hrs_stop_margin': 0.02,
            'hrs_min_cycles_per_alert': 1,
            'hrs_max_cycles_per_alert': 10,
            'hrs_peak_improvement_epsilon': 0.001,
            'hrs_peak_max_moves': 10,
            'hrs_candidate_count': 15,
            'hrs_visit_count': 10, 'hrs_speed': 5.0,
            'hrs_update_seconds': 50.0,
            'hrs_combination_time_limit': 5.0,
            'hrs_planner_mode': 'reward_ordered_exact',
            'hazard_threshold': 0.2,
            'history_top_k': 3,
            'history_merge_radius': 5.0,
            'lrs_history_replace_radius': 5.0,
            'lrs_priority_candidate_count': 15,
            'lrs_priority_count': 6,
            'lrs_route_length_ratio': 1.10,
            'lrs_reward_recurrence_weight': 0.40,
            'lrs_reward_severity_weight': 0.20,
            'lrs_reward_staleness_weight': 0.25,
            'lrs_reward_uncertainty_weight': 0.15,
            'history_recent_alpha': 0.5,
            'history_event_half_life': 10.0,
            'history_event_kernel_sigma': 5.0,
            'history_file': '~/.ros/icir_cleanroom/gas_history.json',
            'source_advance_timeout_seconds': 5.0,
        }
        for name, default in params.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)
        self._validate_parameters()

        self.create_subscription(
            Path, '/gas_mapping/lrs/route', self.path_callback,
            transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_mapping/gmrf_domain', self.map_callback,
            transient_qos())
        self.create_subscription(
            PoseStamped, '/gas_sensor/sensor_pose', self.pose_callback, 30)
        self.create_subscription(
            Float64, '/gas_mapping/sensor_concentration',
            self.value_callback, 30)

        self.estimate_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/estimate', transient_qos())
        self.estimate_log_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/estimate_log', transient_qos())
        self.variance_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/variance', transient_qos())
        self.reward_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/reward', transient_qos())
        self.poses_pub = self.create_publisher(
            PoseArray, '/gas_mapping/measurements/poses', transient_qos())
        self.values_pub = self.create_publisher(
            Float32MultiArray, '/gas_mapping/measurements/values',
            transient_qos())
        self.lrs_status_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/status', transient_qos())
        self.lrs_status_log_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/status_log', transient_qos())
        self.lrs_active_route_pub = self.create_publisher(
            Path, '/gas_mapping/lrs/active_route', transient_qos())
        self.lrs_reward_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/lrs/reward', transient_qos())
        self.lrs_priority_candidates_pub = self.create_publisher(
            Marker, '/gas_mapping/lrs/priority_candidates', transient_qos())
        self.lrs_priority_route_pub = self.create_publisher(
            Path, '/gas_mapping/lrs/priority_route', transient_qos())
        self.history_map_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/history/concentration',
            transient_qos())
        self.history_log_map_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/history/concentration_log',
            transient_qos())
        self.history_recent_log_map_pub = self.create_publisher(
            OccupancyGrid, '/gas_mapping/history/recent_log', transient_qos())
        self.history_confirmed_peaks_pub = self.create_publisher(
            Marker, '/gas_mapping/history/confirmed_peaks', transient_qos())
        self.history_revisit_pub = self.create_publisher(
            Marker, '/gas_mapping/history/revisit', transient_qos())
        self.lrs_lap_pub = self.create_publisher(
            Int32, '/gas_mapping/lrs/lap', transient_qos())
        self.hazard_pub = self.create_publisher(
            Bool, '/gas_mapping/hazard_state', transient_qos())
        self.hrs_route_pub = self.create_publisher(
            Path, '/gas_mapping/hrs/route', transient_qos())
        self.hrs_candidates_pub = self.create_publisher(
            Marker, '/gas_mapping/hrs/candidates', transient_qos())
        self.hrs_status_pub = self.create_publisher(
            Marker, '/gas_mapping/hrs/status', transient_qos())
        self.phase_pub = self.create_publisher(
            String, '/gas_mapping/phase', transient_qos())
        self.create_service(
            Trigger, '/gas_mapping/history/clear', self.clear_history)
        self.source_advance_client = self.create_client(
            Trigger, '/gas_mapping/source/advance')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.map_msg = None
        self.gmrf = None
        self.history = None
        self.base_lrs_goals = []
        self.lrs_goals = []
        self.lrs_status_values = []
        self.active_history_regions = []
        self.lrs_reward_values = None
        self.lrs_reward_points = []
        self.lrs_priority_candidates = []
        self.lrs_priority_points = []
        self.latest_pose = None
        self.measurement_pose = None
        self.dwell_values = []
        self.measuring = False
        self.dwell_timer = None
        self.measured_poses = []
        self.measured_values = []
        self.sampled_variables = set()
        self.hrs_values = {}
        self.failure_counts = {}
        self.unreachable_variables = set()

        self.phase = 'WAITING'
        self.current_index = 0
        self.retry = 0
        self.returning = False
        self.active_hrs_route = None
        self.hrs_cycles = 0
        self.hrs_batch_successes = 0
        self.hrs_cycle_started_ns = None
        self.hrs_cycles_in_alert = 0
        self.lrs_lap = 0
        self.lap_max_concentration = 0.0
        self.event_best_observed = 0.0
        self.event_best_variable = None
        self.lap_hazard_detected = False
        self.peak_anchor_variable = None
        self.peak_move_count = 0
        self.peak_trigger_reason = ''
        self.peak_returning = False
        self.hrs_gmrf_dirty = False
        self.history_snapshot_pending = False
        self.source_advance_future = None
        self.source_advance_timeout_timer = None
        self.source_transition_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.source_transition_reason = ''
        self.current_event_id = ''
        self.planner_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='hrs_p2')
        self.planning_future = None
        self.planning_kind = None
        self.lrs_planning_context = None
        self.start_timer = self.create_timer(1.0, self.try_start)
        self.planning_timer = self.create_timer(0.2, self.poll_planning)
        self.publish_phase('WAITING')
        self.hazard_pub.publish(Bool(data=False))
        self.get_logger().info(
            'LRS route, GMRF domain, sensor pose와 Nav2를 기다리는 중...')

    def _validate_parameters(self):
        if min(float(self.lrs_dwell_seconds),
               float(self.hrs_dwell_seconds)) < 0.0:
            raise ValueError('dwell time must be non-negative')
        if int(self.hrs_candidate_count) <= 0 or int(self.hrs_visit_count) <= 0:
            raise ValueError('HRS candidate and visit counts must be positive')
        if not 0.0 <= float(self.hazard_threshold) <= 1.0:
            raise ValueError('hazard_threshold must be in [0, 1]')
        if int(self.history_top_k) <= 0:
            raise ValueError('history_top_k must be positive')
        if (int(self.lrs_priority_candidate_count) <= 0 or
                int(self.lrs_priority_count) <= 0):
            raise ValueError('LRS priority candidate/count must be positive')
        if (int(self.lrs_priority_count) >
                int(self.lrs_priority_candidate_count)):
            raise ValueError(
                'lrs_priority_count cannot exceed candidate count')
        if (not math.isfinite(float(self.lrs_route_length_ratio)) or
                float(self.lrs_route_length_ratio) < 1.0):
            raise ValueError(
                'lrs_route_length_ratio must be finite and at least 1')
        reward_weights = [
            float(self.lrs_reward_recurrence_weight),
            float(self.lrs_reward_severity_weight),
            float(self.lrs_reward_staleness_weight),
            float(self.lrs_reward_uncertainty_weight),
        ]
        if (not all(math.isfinite(value) and value >= 0.0
                    for value in reward_weights) or
                sum(reward_weights) <= 0.0):
            raise ValueError(
                'LRS reward weights must be finite, non-negative, and '
                'have a positive sum')
        if (not 0.0 < float(self.history_recent_alpha) <= 1.0 or
                not math.isfinite(float(self.history_recent_alpha))):
            raise ValueError('history_recent_alpha must be in (0, 1]')
        if (not math.isfinite(float(self.history_event_half_life)) or
                float(self.history_event_half_life) <= 0.0 or
                not math.isfinite(float(self.history_event_kernel_sigma)) or
                float(self.history_event_kernel_sigma) <= 0.0):
            raise ValueError(
                'history event half-life and kernel sigma must be positive')
        if float(self.hrs_ucb_k) < 0.0:
            raise ValueError('hrs_ucb_k must be non-negative')
        if not 0.0 <= float(self.hrs_stop_margin) <= 1.0:
            raise ValueError('hrs_stop_margin must be in [0, 1]')
        if not 0.0 < float(self.hrs_peak_improvement_epsilon) <= 1.0:
            raise ValueError(
                'hrs_peak_improvement_epsilon must be in (0, 1]')
        if int(self.hrs_peak_max_moves) <= 0:
            raise ValueError('hrs_peak_max_moves must be positive')
        if (int(self.hrs_min_cycles_per_alert) <= 0 or
                int(self.hrs_max_cycles_per_alert) <
                int(self.hrs_min_cycles_per_alert)):
            raise ValueError(
                'HRS max cycles must be at least the positive minimum')
        if (float(self.history_merge_radius) < 0.0 or
                float(self.lrs_history_replace_radius) < 0.0):
            raise ValueError('history radii must be non-negative')
        if (float(self.hrs_speed) <= 0.0 or
                float(self.hrs_update_seconds) <= 0.0):
            raise ValueError('HRS speed and update period must be positive')
        if not 0.0 < float(self.gabp_damping) <= 1.0:
            raise ValueError('GaBP damping must be in (0, 1]')
        if not 0.0 < float(self.gabp_retry_damping) <= 1.0:
            raise ValueError('GaBP retry damping must be in (0, 1]')
        if str(self.hrs_planner_mode) not in PLANNER_MODES:
            raise ValueError(
                f'hrs_planner_mode must be one of {PLANNER_MODES}')
        source_timeout = float(self.source_advance_timeout_seconds)
        if not math.isfinite(source_timeout) or source_timeout <= 0.0:
            raise ValueError('source_advance_timeout_seconds must be positive')

    def destroy_node(self):
        self.planner_pool.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def path_callback(self, msg):
        if self.phase == 'WAITING' and len(msg.poses) >= 4:
            self.base_lrs_goals = list(msg.poses[:-1])
            self.get_logger().info(
                f'LRS closed route 수신: {len(self.base_lrs_goals)} points')

    def map_callback(self, msg):
        if self.gmrf is not None:
            return
        self.map_msg = copy.deepcopy(msg)
        self.gmrf = GmrfGrid(
            msg, self.smoothness_precision, self.background_precision,
            self.background_mean, self.observation_variance_scale,
            self.observation_variance_floor, self.cardinal_weight,
            self.diagonal_weight)
        self.history = GasHistoryStore(
            self.history_file, msg, float(self.hazard_threshold),
            recent_alpha=float(self.history_recent_alpha))
        try:
            loaded = self.history.load()
            if loaded:
                self.get_logger().info(
                    f'가스 이력 {len(self.history.records)}개 셀 복원: '
                    f'{self.history.path}, '
                    f'distribution_count={self.history.history_count}')
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.get_logger().warning(
                f'가스 이력 파일을 사용할 수 없어 빈 이력으로 시작합니다: '
                f'{error}')
            self.history.reset()
        self.update_gmrf('initialization')
        self.publish_history()
        self.publish_hrs_status()
        self.get_logger().info(
            f'GMRF 생성: {len(self.gmrf.solution)} free variables')

    def pose_callback(self, msg):
        self.latest_pose = copy.deepcopy(msg)

    def value_callback(self, msg):
        if self.measuring:
            self.dwell_values.append(float(msg.data))

    def try_start(self):
        if (self.phase != 'WAITING' or not self.base_lrs_goals or
                self.gmrf is None or self.latest_pose is None or
                not self.nav_client.server_is_ready()):
            return
        self.start_timer.cancel()
        self.start_lrs_lap('initial patrol')

    def start_lrs_lap(self, reason):
        if not self.base_lrs_goals or self.latest_pose is None:
            return
        self.lrs_lap += 1
        self.history_snapshot_pending = True
        self.current_index = 0
        self.retry = 0
        self.returning = False
        self.lap_max_concentration = 0.0
        self.event_best_observed = 0.0
        self.event_best_variable = None
        self.lap_hazard_detected = False
        self.peak_anchor_variable = None
        self.peak_move_count = 0
        self.peak_trigger_reason = ''
        self.peak_returning = False
        self.hrs_gmrf_dirty = False
        self.sampled_variables.clear()
        self.hrs_values.clear()
        self.failure_counts.clear()
        self.unreachable_variables.clear()
        self.hrs_cycles_in_alert = 0
        if not self.current_event_id:
            self.current_event_id = f'controller-source-{time.time_ns()}'
        self.publish_hrs_status()

        self.gmrf.reset_observations()
        if not self.update_gmrf(f'LRS lap {self.lrs_lap} reset'):
            self.get_logger().warning(
                '새 LRS 회차의 background-prior GMRF 초기화가 '
                '수렴하지 않았습니다')

        self.active_history_regions = self.history.confirmed_regions(
            top_k=int(self.history_top_k),
            merge_radius=float(self.history_merge_radius),
            event_half_life=float(self.history_event_half_life))
        base_points = [
            (goal.pose.position.x, goal.pose.position.y)
            for goal in self.base_lrs_goals]
        history_points = [
            (record['x'], record['y'])
            for record in self.active_history_regions]
        current_xy = (
            self.latest_pose.pose.position.x,
            self.latest_pose.pose.position.y)
        adjusted, replacements, additions = history_adjusted_cycle(
            base_points, history_points,
            float(self.lrs_history_replace_radius), current_xy)
        points = self.prepare_adaptive_lrs_points(adjusted)
        if self.planning_future is not None:
            raise RuntimeError('another route-planning job is already active')
        self.lrs_planning_context = {
            'reason': str(reason), 'base_points': base_points,
            'history_points': history_points,
            'replacements': replacements, 'additions': additions,
            'points': points, 'current_xy': current_xy,
            'original_baseline_length': self.closed_cycle_reference_length(
                base_points, current_xy),
        }
        self.lrs_goals = []
        self.lrs_status_values = []
        self.lrs_priority_points = []
        self.publish_phase('LRS_PLANNING')
        self.hazard_pub.publish(Bool(data=False))
        self.lrs_lap_pub.publish(Int32(data=self.lrs_lap))
        self.publish_lrs_active_route()
        self.publish_lrs_reward()
        self.publish_lrs_priority_candidates()
        self.publish_lrs_priority_route()
        self.publish_lrs_status()
        self.publish_history()
        self.get_logger().info(
            f'=== LRS lap {self.lrs_lap} 경로 계산 시작 ({reason}): '
            f'base={len(base_points)}, active={len(points)}, '
            f'history_regions={len(history_points)}, '
            f'replaced={len(replacements)}, added={len(additions)} ===')
        if replacements:
            self.get_logger().info(
                'LRS 역사 위치 대체: ' + str([
                    (tuple(round(value, 3) for value in old),
                     tuple(round(value, 3) for value in new),
                     round(separation, 3))
                    for old, new, separation in replacements]))
        if additions:
            self.get_logger().info(
                'LRS 역사 위치 추가: ' + str([
                    tuple(round(value, 3) for value in point)
                    for point in additions]))
        self.planning_kind = 'LRS'
        self.planning_future = self.planner_pool.submit(
            solve_lrs_priority_route, points, current_xy,
            int(self.lrs_priority_candidate_count),
            int(self.lrs_priority_count),
            float(self.lrs_route_length_ratio))

    def prepare_adaptive_lrs_points(self, adjusted_points):
        """Calculate active-set LRS rewards before threaded route solving."""
        latest = self.history.latest_confirmed_peak()
        mandatory_variable = None
        if latest is not None:
            mandatory_variable = self.gmrf.variable_at(
                int(latest['row']), int(latest['col']))

        active = []
        active_index_by_variable = {}
        for x, y in adjusted_points:
            variable = int(self.gmrf.nearest_variable(float(x), float(y)))
            if variable in active_index_by_variable:
                row, col = self.gmrf.var_cells[variable]
                self.get_logger().warning(
                    f'중복 LRS GMRF 셀 ({int(row)},{int(col)})을 '
                    '이번 활성 경로에서 한 번만 유지합니다')
                if variable == mandatory_variable:
                    active[active_index_by_variable[variable]] = (
                        variable, float(x), float(y))
                continue
            active_index_by_variable[variable] = len(active)
            active.append((variable, float(x), float(y)))
        if len(active) < 2:
            raise ValueError('adaptive LRS requires at least two unique cells')

        active_cells = np.asarray([
            self.gmrf.var_cells[variable] for variable, _, _ in active],
            dtype=np.int64)
        now = time.time()
        staleness_horizon = self.history.lrs_staleness_horizon(
            active_cells, now)
        reward_data = self.history.lrs_rewards(
            active_cells, now,
            recurrence_weight=float(self.lrs_reward_recurrence_weight),
            severity_weight=float(self.lrs_reward_severity_weight),
            staleness_weight=float(self.lrs_reward_staleness_weight),
            uncertainty_weight=float(self.lrs_reward_uncertainty_weight),
            event_half_life=float(self.history_event_half_life),
            kernel_sigma=float(self.history_event_kernel_sigma),
            staleness_horizon=staleness_horizon)
        display_reward_data = self.history.lrs_rewards(
            self.gmrf.var_cells, now,
            recurrence_weight=float(self.lrs_reward_recurrence_weight),
            severity_weight=float(self.lrs_reward_severity_weight),
            staleness_weight=float(self.lrs_reward_staleness_weight),
            uncertainty_weight=float(self.lrs_reward_uncertainty_weight),
            event_half_life=float(self.history_event_half_life),
            kernel_sigma=float(self.history_event_kernel_sigma),
            staleness_horizon=staleness_horizon)
        self.lrs_reward_values = np.asarray(
            display_reward_data['reward'], dtype=float)

        points = []
        for index, (variable, x, y) in enumerate(active):
            row, col = self.gmrf.var_cells[variable]
            point = LrsRewardPoint(
                variable=variable, row=int(row), col=int(col),
                x=float(x), y=float(y),
                reward=float(reward_data['reward'][index]),
                recurrence=float(reward_data['recurrence'][index]),
                severity=float(reward_data['severity'][index]),
                staleness=float(reward_data['staleness'][index]),
                uncertainty=float(reward_data['uncertainty'][index]),
                mandatory=(variable == mandatory_variable))
            points.append(point)

        ranked = sorted(points, key=lambda point: (
            not point.mandatory, -point.reward, point.row, point.col,
            point.variable))
        self.lrs_priority_candidates = ranked[
            :int(self.lrs_priority_candidate_count)]
        for point in self.lrs_priority_candidates:
            self.get_logger().info(
                f'LRS reward cell=({point.row},{point.col}), '
                f'total={point.reward:.4f}, '
                f'recurrence={point.recurrence:.4f}, '
                f'severity={point.severity:.4f}, '
                f'staleness={point.staleness:.4f}, '
                f'uncertainty={point.uncertainty:.4f}, '
                f'mandatory={point.mandatory}')
        return points

    def finish_lrs_route_planning(self, route_plan, context):
        if self.phase != 'LRS_PLANNING':
            self.get_logger().warning(
                f'현재 phase={self.phase}이므로 완료된 LRS 경로를 '
                '적용하지 않습니다')
            return
        self.lrs_reward_points = list(route_plan.points)
        self.lrs_priority_points = list(route_plan.priority_points)
        self.lrs_goals = [
            self.make_pose(point.x, point.y) for point in route_plan.points]
        self.lrs_status_values = [None] * len(self.lrs_goals)
        self.current_index = 0
        self.retry = 0
        self.returning = False

        self.publish_phase('LRS')
        self.publish_lrs_active_route()
        self.publish_lrs_priority_route()
        self.publish_lrs_status()
        base_points = context['base_points']
        history_points = context['history_points']
        replacements = context['replacements']
        additions = context['additions']
        self.get_logger().info(
            f'=== LRS lap {self.lrs_lap} 시작 ({context["reason"]}): '
            f'base={len(base_points)}, active={len(self.lrs_goals)}, '
            f'history_regions={len(history_points)}, '
            f'replaced={len(replacements)}, added={len(additions)}, '
            f'priority={route_plan.priority_count}, '
            f'route={route_plan.length:.3f}m, '
            f'active_baseline={route_plan.baseline_length:.3f}m, '
            f'original_p1={context["original_baseline_length"]:.3f}m, '
            f'({route_plan.length_ratio:.3f}x), '
            f'fallback={route_plan.fallback} ===')
        self.get_logger().info(
            f'LRS reward 경로 계산: status={route_plan.status}, '
            f'candidates={len(self.lrs_priority_candidates)}, '
            f'total_combinations={route_plan.total_combinations}, '
            f'routes_solved={route_plan.evaluated_routes}, '
            f'feasible={route_plan.feasible_routes}, '
            f'priority_reward={route_plan.reward:.6f}, '
            f'planning={route_plan.planning_seconds:.4f}s, '
            f'priority_cells='
            f'{[(point.row, point.col) for point in route_plan.priority_points]}')
        self.send_current_goal()

    @staticmethod
    def closed_cycle_reference_length(points, start_xy):
        if not points:
            return 0.0
        cycle = 0.0
        for index, first in enumerate(points):
            second = points[(index + 1) % len(points)]
            cycle += math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]))
        entry = min(math.hypot(
            float(point[0]) - float(start_xy[0]),
            float(point[1]) - float(start_xy[1])) for point in points)
        return cycle + entry

    @staticmethod
    def make_pose(x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        return pose

    def publish_phase(self, phase):
        self.phase = phase
        self.phase_pub.publish(String(data=phase))
        self.get_logger().info(f'Gas mapping phase: {phase}')

    @staticmethod
    def is_hrs_navigation_phase(phase):
        return phase in ('HRS_NAVIGATION', 'HRS_PEAK_CONFIRMATION')

    def record_event_measurement(self, variable, value):
        variable = int(variable)
        value = float(value)
        self.sampled_variables.add(variable)
        self.hrs_values[variable] = value
        best = min(
            self.hrs_values,
            key=lambda item: (
                -float(self.hrs_values[item]),
                int(self.gmrf.var_cells[item][0]),
                int(self.gmrf.var_cells[item][1])))
        self.event_best_variable = int(best)
        self.event_best_observed = float(self.hrs_values[best])

    def active_target(self):
        if self.phase == 'LRS':
            if self.returning:
                return self.lrs_goals[0]
            if self.current_index < len(self.lrs_goals):
                return self.lrs_goals[self.current_index]
        if (self.is_hrs_navigation_phase(self.phase) and
                self.active_hrs_route is not None and
                self.current_index < len(self.active_hrs_route.cells)):
            cell = self.active_hrs_route.cells[self.current_index]
            target = PoseStamped()
            target.header.frame_id = 'map'
            target.pose.position.x = cell.x
            target.pose.position.y = cell.y
            target.pose.orientation.w = 1.0
            return target
        return None

    def send_current_goal(self):
        target = self.active_target()
        if target is None:
            if self.phase == 'LRS':
                self.finish_lrs_navigation()
            elif self.phase == 'HRS_NAVIGATION':
                self.finish_hrs_cycle()
            elif self.phase == 'HRS_PEAK_CONFIRMATION':
                self.finish_peak_confirmation_batch()
            return

        goal = NavigateToPose.Goal()
        goal.pose = copy.deepcopy(target)
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.orientation = self.goal_orientation(target)
        if self.phase == 'LRS':
            label = ('시작 LRS 포인트 복귀' if self.returning else
                     f'LRS [{self.current_index + 1}/{len(self.lrs_goals)}]')
        elif self.phase == 'HRS_PEAK_CONFIRMATION':
            if self.peak_returning:
                label = 'HRS confirmed peak return'
            else:
                label = (
                    f'HRS peak move {self.peak_move_count} '
                    f'[{self.current_index + 1}/'
                    f'{len(self.active_hrs_route.cells)}]')
        else:
            label = (
                f'HRS cycle {self.hrs_cycles + 1} '
                f'[{self.current_index + 1}/'
                f'{len(self.active_hrs_route.cells)}]')
        self.get_logger().info(
            f'{label} -> ({target.pose.position.x:.2f}, '
            f'{target.pose.position.y:.2f})')
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response)

    def goal_orientation(self, target):
        following = None
        if self.phase == 'LRS' and not self.returning and self.lrs_goals:
            following = self.lrs_goals[
                (self.current_index + 1) % len(self.lrs_goals)]
        elif (self.is_hrs_navigation_phase(self.phase) and
              self.active_hrs_route and
              self.current_index + 1 < len(self.active_hrs_route.cells)):
            cell = self.active_hrs_route.cells[self.current_index + 1]
            following = PoseStamped()
            following.pose.position.x = cell.x
            following.pose.position.y = cell.y
        pose = copy.deepcopy(target.pose.orientation)
        if following is None:
            pose.w = 1.0
            return pose
        yaw = math.atan2(
            following.pose.position.y - target.pose.position.y,
            following.pose.position.x - target.pose.position.x)
        pose.x = pose.y = 0.0
        pose.z = math.sin(yaw / 2.0)
        pose.w = math.cos(yaw / 2.0)
        return pose

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
        if self.phase == 'LRS' and self.returning:
            self.finish_lrs_navigation()
            return
        if self.phase == 'HRS_PEAK_CONFIRMATION' and self.peak_returning:
            self.finish_peak_return()
            return
        self.measurement_pose = copy.deepcopy(self.latest_pose)
        self.dwell_values = []
        self.measuring = True
        dwell = (self.lrs_dwell_seconds if self.phase == 'LRS'
                 else self.hrs_dwell_seconds)
        self.dwell_timer = self.create_timer(float(dwell), self.finish_dwell)

    def finish_dwell(self):
        self.dwell_timer.cancel()
        self.dwell_timer = None
        self.measuring = False
        if not self.dwell_values:
            self.get_logger().warning(
                '농도 표본이 없어 이 포인트를 미측정으로 남깁니다')
            if self.is_hrs_navigation_phase(self.phase):
                self.record_hrs_failure()
        else:
            value = float(sum(self.dwell_values) / len(self.dwell_values))
            pose = self.measurement_pose.pose
            self.measured_poses.append(copy.deepcopy(pose))
            self.measured_values.append(value)
            row, col = self.gmrf.set_observation(
                pose.position.x, pose.position.y, value)
            variable = self.gmrf.variable_at(int(row), int(col))
            self.history.update(
                int(row), int(col), value, self.phase, time.time())
            self.publish_history()
            self.record_event_measurement(variable, value)
            if self.phase == 'LRS':
                self.lrs_status_values[self.current_index] = value
                self.lap_max_concentration = max(
                    self.lap_max_concentration, value)
                if value >= float(self.hazard_threshold):
                    if not self.lap_hazard_detected:
                        self.get_logger().warning(
                            f'위험 가스 감지: value={value:.4f} >= '
                            f'{float(self.hazard_threshold):.4f}; '
                            '현재 LRS 순찰 완료 후 HRS로 전환합니다')
                    self.lap_hazard_detected = True
                    self.hazard_pub.publish(Bool(data=True))
                self.update_gmrf('LRS measurement')
            else:
                planned = self.active_hrs_route.cells[self.current_index]
                if variable == planned.variable:
                    self.failure_counts.pop(planned.variable, None)
                    self.hrs_batch_successes += 1
                else:
                    self.get_logger().warning(
                        f'목표 셀 ({planned.row},{planned.col})과 실제 측정 '
                        f'셀 ({int(row)},{int(col)})이 달라 목표를 '
                        '미측정으로 유지합니다')
                    self.record_hrs_failure()
                self.hrs_gmrf_dirty = True
                if self.update_gmrf(
                        f'{self.phase} measurement '
                        f'[{self.current_index + 1}/'
                        f'{len(self.active_hrs_route.cells)}]',
                        compare_with_cg=False):
                    self.hrs_gmrf_dirty = False
                else:
                    self.get_logger().warning(
                        'HRS 지점별 GaBP 갱신을 다음 측정 또는 '
                        '배치 종료 시 재시도합니다')
            self.publish_measurements()
            self.get_logger().info(
                f'측정 완료: phase={self.phase}, '
                f'pose=({pose.position.x:.3f},{pose.position.y:.3f}), '
                f'mean={value:.4f}, samples={len(self.dwell_values)}, '
                f'cell=({row},{col})')
        self.advance_after_target()

    def navigation_failed(self, reason):
        self.retry += 1
        if self.retry <= int(self.max_retries):
            self.get_logger().warning(
                f'이동 실패({reason}), 재시도 '
                f'{self.retry}/{self.max_retries}')
            self.send_current_goal()
            return
        self.retry = 0
        self.get_logger().warning(
            f'이동 실패({reason}), 미측정으로 남기고 다음 포인트 진행')
        if self.phase == 'LRS' and self.returning:
            self.finish_lrs_navigation()
            return
        if self.phase == 'HRS_PEAK_CONFIRMATION' and self.peak_returning:
            self.return_to_lrs(
                f'peak_unconfirmed: failed to return to confirmed peak '
                f'({reason})')
            return
        if self.phase == 'HRS_PEAK_CONFIRMATION':
            cell = self.active_hrs_route.cells[self.current_index]
            self.unreachable_variables.add(cell.variable)
            self.failure_counts[cell.variable] = int(self.max_cell_failures)
            self.get_logger().error(
                f'HRS 최고점 이웃 ({cell.row},{cell.col})가 '
                f'{int(self.max_retries) + 1}회 이동 실패하여 '
                '국소 최고점을 확인할 수 없습니다')
        elif self.phase == 'HRS_NAVIGATION':
            self.record_hrs_failure()
        self.advance_after_target()

    def record_hrs_failure(self):
        cell = self.active_hrs_route.cells[self.current_index]
        count = self.failure_counts.get(cell.variable, 0) + 1
        self.failure_counts[cell.variable] = count
        if count >= int(self.max_cell_failures):
            self.unreachable_variables.add(cell.variable)
            self.get_logger().error(
                f'HRS cell ({cell.row},{cell.col})가 {count}회 실패하여 '
                '도달 불가로 기록되었습니다')

    def advance_after_target(self):
        self.current_index += 1
        if self.phase == 'LRS':
            self.publish_lrs_status()
            if self.current_index >= len(self.lrs_goals):
                self.returning = True
            self.send_current_goal()
        elif self.is_hrs_navigation_phase(self.phase):
            self.publish_hrs_status()
            self.send_current_goal()

    def finish_lrs_navigation(self):
        self.returning = False
        self.get_logger().info(
            f'=== LRS lap {self.lrs_lap} 완료: '
            f'max={self.lap_max_concentration:.4f}, '
            f'threshold={float(self.hazard_threshold):.4f}, '
            f'hazard={self.lap_hazard_detected} ===')
        if self.lap_hazard_detected:
            self.persist_history('hazardous LRS measurements')
            self.hrs_cycles_in_alert = 0
            self.start_hrs_planning()
        else:
            self.commit_history_snapshot(
                f'non-hazardous LRS lap {self.lrs_lap}')
            self.start_lrs_lap('no hazardous concentration detected')

    def available_variables(self):
        all_variables = set(range(len(self.gmrf.var_cells)))
        return (all_variables - self.sampled_variables -
                self.unreachable_variables)

    def build_candidates(self):
        candidates = []
        potentials = normalized_ucb(
            self.gmrf.solution, self.gmrf.variance,
            float(self.hrs_ucb_k))
        for variable in self.available_variables():
            row, col = self.gmrf.var_cells[variable]
            x, y = self.gmrf.cell_center(variable)
            mean = float(self.gmrf.solution[variable])
            variance = float(self.gmrf.variance[variable])
            candidates.append(HrsCell(
                variable=variable, row=int(row), col=int(col), x=x, y=y,
                reward=float(potentials[variable]),
                mean=mean, variance=variance))
        candidates.sort(key=lambda cell: (-cell.reward, cell.row, cell.col))
        return candidates[:int(self.hrs_candidate_count)]

    def start_hrs_planning(self):
        if self.planning_future is not None:
            return
        available = self.available_variables()
        if not available:
            reason = ('all reachable cells sampled' if not self.unreachable_variables
                      else 'no reachable unsampled cells remain')
            self.start_peak_confirmation(reason)
            return
        candidates = self.build_candidates()
        self.publish_candidates(candidates)
        current_xy = (
            self.latest_pose.pose.position.x,
            self.latest_pose.pose.position.y)
        max_visits = min(int(self.hrs_visit_count), len(candidates))
        self.publish_phase('HRS_PLANNING')
        self.get_logger().info(
            f'HRS cycle {self.hrs_cycles + 1}: Q_u={len(available)}, '
            f'candidates={len(candidates)}, '
            f'ucb_k={float(self.hrs_ucb_k):.3f}, '
            f'visit_count={max_visits}, planner={str(self.hrs_planner_mode)}, '
            f'top_M='
            f'{[(cell.row, cell.col, round(cell.reward, 6)) for cell in candidates]}')
        self.planning_kind = 'HRS'
        self.planning_future = self.planner_pool.submit(
            self.plan_with_fallback, candidates, current_xy, max_visits)

    def plan_with_fallback(self, candidates, current_xy, max_visits):
        attempts = []
        for visit_count in range(max_visits, 0, -1):
            plan = solve_hrs_p2(
                candidates, current_xy,
                visit_count=visit_count,
                speed=float(self.hrs_speed),
                update_seconds=float(self.hrs_update_seconds),
                dwell_seconds=float(self.hrs_dwell_seconds),
                combination_time_limit=float(
                    self.hrs_combination_time_limit),
                planner_mode=str(self.hrs_planner_mode))
            attempts.append((visit_count, plan))
            if plan is not None:
                return plan, attempts
        return None, attempts

    def poll_planning(self):
        if self.planning_future is None or not self.planning_future.done():
            return
        future = self.planning_future
        planning_kind = self.planning_kind
        self.planning_future = None
        self.planning_kind = None
        if planning_kind == 'LRS':
            context = self.lrs_planning_context
            self.lrs_planning_context = None
            if context is None:
                self.get_logger().error(
                    'LRS planning context가 없어 완료 결과를 버립니다')
                return
            try:
                route_plan = future.result()
            except Exception as error:
                self.get_logger().error(
                    f'LRS reward route worker failed: {error!r}; '
                    '최단 coverage fallback을 사용합니다')
                route_plan = lrs_coverage_fallback(
                    context['points'], context['current_xy'])
            self.finish_lrs_route_planning(route_plan, context)
            return
        try:
            plan, attempts = future.result()
        except Exception as error:  # keep ROS alive and report worker failures
            self.get_logger().error(f'HRS P2 worker failed: {error!r}')
            self.start_peak_confirmation('P2 worker failure')
            return
        for visit_count, result in attempts:
            if result is None:
                self.get_logger().warning(
                    f'HRS P2 T={visit_count}: feasible route 없음')
        if plan is None:
            self.start_peak_confirmation('no feasible P2 route')
            return
        self.active_hrs_route = plan
        self.current_index = 0
        self.retry = 0
        self.hrs_batch_successes = 0
        self.hrs_cycle_started_ns = self.get_clock().now().nanoseconds
        self.publish_hrs_route(plan)
        self.publish_phase('HRS_NAVIGATION')
        self.get_logger().info(
            f'HRS P2 선택: T={plan.visit_count}, '
            f'mode={str(self.hrs_planner_mode)}, solver={plan.solver}, '
            f'total_combinations={plan.total_combinations}, '
            f'routes_solved={plan.evaluated_routes}, '
            f'lower_bound_pruned={plan.pruned_combinations}, '
            f'feasible={plan.feasible_combinations}, reward={plan.reward:.6f}, '
            f'length={plan.length:.3f}m, expected={plan.expected_seconds:.3f}s, '
            f'planning={plan.planning_seconds:.3f}s, '
            f'status={plan.status}, gap={plan.gap:.6f}, cuts={plan.cuts}, '
            f'selected={[(cell.row, cell.col) for cell in plan.cells]}')
        self.send_current_goal()

    def finish_hrs_cycle(self):
        actual_seconds = (
            (self.get_clock().now().nanoseconds - self.hrs_cycle_started_ns)
            * 1.0e-9)
        deadline = float(self.hrs_update_seconds)
        deadline_text = ('DEADLINE MISS' if actual_seconds > deadline
                         else 'on time')
        self.hrs_cycles += 1
        self.hrs_cycles_in_alert += 1
        self.get_logger().info(
            f'=== HRS cycle {self.hrs_cycles} 완료: '
            f'success={self.hrs_batch_successes}/'
            f'{len(self.active_hrs_route.cells)}, '
            f'expected={self.active_hrs_route.expected_seconds:.3f}s, '
            f'actual={actual_seconds:.3f}s, {deadline_text} ===')
        map_updated = self.finalize_hrs_gmrf_batch(
            f'HRS cycle {self.hrs_cycles}')
        self.publish_hrs_status()
        self.persist_history(f'HRS cycle {self.hrs_cycles}')
        self.active_hrs_route = None
        if not map_updated:
            self.get_logger().warning(
                f'HRS 수렴 판정 보류: '
                f'alert_cycle={self.hrs_cycles_in_alert}, '
                'dirty GaBP map could not be recovered')
            if self.available_variables():
                self.start_hrs_planning()
            else:
                self.return_to_lrs(
                    'peak_unconfirmed: GMRF update failed and no '
                    'unvisited cell remains')
            return
        converged, detail = self.evaluate_hrs_stop()
        self.get_logger().info(detail)
        decision = self.hrs_stop_decision(
            self.hrs_cycles_in_alert, converged,
            int(self.hrs_min_cycles_per_alert),
            int(self.hrs_max_cycles_per_alert))
        if decision == 'converged':
            self.start_peak_confirmation(
                'no reachable unvisited cell has a higher concentration '
                'potential')
        elif decision == 'max_cycles':
            self.start_peak_confirmation(
                f'HRS maximum {self.hrs_max_cycles_per_alert} cycles reached '
                'without convergence')
        else:
            self.start_hrs_planning()

    @staticmethod
    def hrs_stop_decision(cycles, converged, minimum_cycles, maximum_cycles):
        if int(cycles) >= int(minimum_cycles) and bool(converged):
            return 'converged'
        if int(cycles) >= int(maximum_cycles):
            return 'max_cycles'
        return None

    def evaluate_hrs_stop(self):
        available = self.available_variables()
        converged, variable, maximum, gap = peak_search_status(
            self.gmrf.solution, self.gmrf.variance, available,
            self.event_best_observed, float(self.hrs_ucb_k),
            float(self.hrs_stop_margin))
        if variable is None:
            return True, (
                f'HRS 수렴 판정: alert_cycle={self.hrs_cycles_in_alert}, '
                f'best_observed={self.event_best_observed:.4f}, '
                'available=0, converged=True')
        row, col = self.gmrf.var_cells[variable]
        x, y = self.gmrf.cell_center(variable)
        return converged, (
            f'HRS 수렴 판정: alert_cycle={self.hrs_cycles_in_alert}, '
            f'best_observed={self.event_best_observed:.4f}, '
            f'best_unvisited_potential={maximum:.4f}, gap={gap:.4f}, '
            f'margin={float(self.hrs_stop_margin):.4f}, '
            f'cell=({int(row)},{int(col)}), '
            f'position=({x:.3f},{y:.3f}), converged={converged}')

    def start_peak_confirmation(self, trigger_reason):
        self.active_hrs_route = None
        if (self.event_best_variable is None or
                self.event_best_variable not in self.hrs_values):
            self.return_to_lrs(
                f'peak_unconfirmed: no measured anchor ({trigger_reason})')
            return
        self.peak_anchor_variable = int(self.event_best_variable)
        self.peak_move_count = 0
        self.peak_trigger_reason = str(trigger_reason)
        self.peak_returning = False
        self.clear_hrs_candidates()
        row, col = self.gmrf.var_cells[self.peak_anchor_variable]
        x, y = self.gmrf.cell_center(self.peak_anchor_variable)
        self.publish_phase('HRS_PEAK_CONFIRMATION')
        self.get_logger().info(
            f'HRS 최고점 확인 시작: trigger={trigger_reason}, '
            f'anchor=({int(row)},{int(col)}), '
            f'position=({x:.3f},{y:.3f}), '
            f'value={self.hrs_values[self.peak_anchor_variable]:.4f}')
        self.plan_peak_confirmation_neighborhood()

    def peak_confirmation_cells(self, variables):
        potentials = normalized_ucb(
            self.gmrf.solution, self.gmrf.variance,
            float(self.hrs_ucb_k))
        cells = []
        for variable in variables:
            row, col = self.gmrf.var_cells[variable]
            x, y = self.gmrf.cell_center(variable)
            cells.append(HrsCell(
                variable=int(variable), row=int(row), col=int(col),
                x=x, y=y, reward=float(potentials[variable]),
                mean=float(self.gmrf.solution[variable]),
                variance=float(self.gmrf.variance[variable])))
        cells.sort(key=lambda cell: (cell.row, cell.col))
        return cells

    def plan_peak_confirmation_neighborhood(self):
        if self.peak_anchor_variable is None:
            self.return_to_lrs('peak_unconfirmed: anchor was lost')
            return
        neighbors = self.gmrf.neighbor_variables(
            self.peak_anchor_variable)
        blocked = [variable for variable in neighbors
                   if variable in self.unreachable_variables]
        if blocked:
            cells = [tuple(int(item) for item in self.gmrf.var_cells[var])
                     for var in blocked]
            self.return_to_lrs(
                f'peak_unconfirmed: unreachable neighbor cells={cells}')
            return
        missing = [variable for variable in neighbors
                   if variable not in self.hrs_values]
        if not missing:
            self.evaluate_peak_confirmation_neighborhood(neighbors)
            return

        cells = self.peak_confirmation_cells(missing)
        current_xy = (
            self.latest_pose.pose.position.x,
            self.latest_pose.pose.position.y)
        plan = solve_peak_confirmation_route(
            cells, current_xy, speed=float(self.hrs_speed),
            dwell_seconds=float(self.hrs_dwell_seconds))
        if plan is None:
            self.return_to_lrs(
                'peak_unconfirmed: neighbor route planning failed')
            return
        self.active_hrs_route = plan
        self.current_index = 0
        self.retry = 0
        self.hrs_batch_successes = 0
        self.publish_hrs_route(plan)
        anchor_row, anchor_col = self.gmrf.var_cells[
            self.peak_anchor_variable]
        self.get_logger().info(
            f'HRS 최고점 이웃 확인: move={self.peak_move_count}, '
            f'anchor=({int(anchor_row)},{int(anchor_col)}), '
            f'unmeasured={len(cells)}, length={plan.length:.3f}m, '
            f'route={[(cell.row, cell.col) for cell in plan.cells]}')
        self.send_current_goal()

    def finish_peak_confirmation_batch(self):
        route = self.active_hrs_route
        self.get_logger().info(
            f'HRS 최고점 이웃 측정 완료: success='
            f'{self.hrs_batch_successes}/{len(route.cells)}')
        map_updated = self.finalize_hrs_gmrf_batch(
            f'HRS peak confirmation move {self.peak_move_count}')
        self.publish_hrs_status()
        self.persist_history(
            f'HRS peak confirmation move {self.peak_move_count}')
        self.active_hrs_route = None
        if not map_updated:
            self.return_to_lrs(
                'peak_unconfirmed: GMRF update failed after neighbor '
                'measurements')
            return
        self.plan_peak_confirmation_neighborhood()

    def evaluate_peak_confirmation_neighborhood(self, neighbors):
        anchor = self.peak_anchor_variable
        try:
            higher = higher_measured_neighbor(
                anchor, neighbors, self.hrs_values, self.gmrf.var_cells,
                float(self.hrs_peak_improvement_epsilon))
        except ValueError as error:
            self.return_to_lrs(f'peak_unconfirmed: {error}')
            return
        if higher is None:
            self.start_peak_return()
            return
        if self.peak_move_count >= int(self.hrs_peak_max_moves):
            row, col = self.gmrf.var_cells[anchor]
            self.return_to_lrs(
                f'peak_unconfirmed: maximum {self.hrs_peak_max_moves} '
                f'center moves reached at ({int(row)},{int(col)})')
            return
        old_row, old_col = self.gmrf.var_cells[anchor]
        new_row, new_col = self.gmrf.var_cells[higher]
        old_value = float(self.hrs_values[anchor])
        new_value = float(self.hrs_values[higher])
        self.peak_anchor_variable = int(higher)
        self.peak_move_count += 1
        self.get_logger().info(
            f'HRS 최고점 중심 이동 {self.peak_move_count}: '
            f'({int(old_row)},{int(old_col)}) {old_value:.4f} -> '
            f'({int(new_row)},{int(new_col)}) {new_value:.4f}')
        self.plan_peak_confirmation_neighborhood()

    def start_peak_return(self):
        anchor = self.peak_anchor_variable
        cell = self.peak_confirmation_cells([anchor])[0]
        current_xy = (
            self.latest_pose.pose.position.x,
            self.latest_pose.pose.position.y)
        plan = solve_peak_confirmation_route(
            [cell], current_xy, speed=float(self.hrs_speed),
            dwell_seconds=0.0)
        if plan is None:
            self.return_to_lrs(
                'peak_unconfirmed: confirmed-peak return planning failed')
            return
        self.peak_returning = True
        self.active_hrs_route = plan
        self.current_index = 0
        self.retry = 0
        self.publish_hrs_route(plan)
        self.get_logger().info(
            f'HRS 국소 최고 셀 복귀: cell=({cell.row},{cell.col}), '
            f'position=({cell.x:.3f},{cell.y:.3f}), '
            f'value={self.hrs_values[anchor]:.4f}, '
            f'distance={plan.length:.3f}m')
        self.send_current_goal()

    def finish_peak_return(self):
        anchor = self.peak_anchor_variable
        row, col = self.gmrf.var_cells[anchor]
        x, y = self.gmrf.cell_center(anchor)
        value = float(self.hrs_values[anchor])
        actual_x = self.latest_pose.pose.position.x
        actual_y = self.latest_pose.pose.position.y
        self.active_hrs_route = None
        self.peak_returning = False
        actual_variable = self.gmrf.nearest_variable(actual_x, actual_y)
        if actual_variable != anchor:
            actual_row, actual_col = self.gmrf.var_cells[actual_variable]
            self.return_to_lrs(
                f'peak_unconfirmed: peak return reached '
                f'({int(actual_row)},{int(actual_col)}) instead of '
                f'({int(row)},{int(col)})')
            return
        self.get_logger().info(
            f'HRS 실측 국소 최고점 도착 및 확정: '
            f'cell=({int(row)},{int(col)}), '
            f'target=({x:.3f},{y:.3f}), '
            f'robot=({actual_x:.3f},{actual_y:.3f}), value={value:.4f}, '
            f'moves={self.peak_move_count}, '
            f'trigger={self.peak_trigger_reason}')
        try:
            inserted = self.history.record_confirmed_peak(
                self.current_event_id, int(row), int(col), value,
                time.time())
        except ValueError as error:
            self.get_logger().error(
                f'확정 최고점 이벤트 기록 실패: {error}')
            inserted = False
        if inserted:
            self.get_logger().info(
                f'독립 가스 이벤트 확정 기록: id={self.current_event_id}, '
                f'cell=({int(row)},{int(col)}), value={value:.4f}')
        else:
            self.get_logger().warning(
                f'이미 기록된 가스 이벤트를 중복 집계하지 않습니다: '
                f'id={self.current_event_id}')
        self.publish_history()
        self.start_source_transition(
            f'peak_confirmed at ({int(row)},{int(col)}) value={value:.4f}')

    def start_source_transition(self, reason):
        if (self.phase == 'SOURCE_TRANSITION' or
                self.source_advance_future is not None):
            self.get_logger().warning(
                '가스원 전환 요청이 이미 진행 중이므로 중복 요청을 '
                f'무시합니다: {reason}')
            return

        self.get_logger().info(
            f'HRS 실측 최고점 확정 후 가스원 전환: {reason}')
        self.commit_history_snapshot(
            f'confirmed hazard event ending after LRS lap {self.lrs_lap}')
        self.peak_returning = False
        self.active_hrs_route = None
        self.publish_empty_hrs_route()
        self.clear_hrs_candidates()
        self.source_transition_reason = str(reason)
        self.publish_phase('SOURCE_TRANSITION')

        if not self.source_advance_client.service_is_ready():
            self.get_logger().warning(
                '/gas_mapping/source/advance 서비스를 사용할 수 없어 '
                '기존 가스원으로 LRS를 계속합니다')
            self.finish_source_transition(False, 'service unavailable')
            return

        try:
            future = self.source_advance_client.call_async(Trigger.Request())
        except Exception as error:  # keep patrol alive on middleware errors
            self.get_logger().warning(
                f'가스원 전환 요청 실패({error!r}); '
                '기존 가스원으로 LRS를 계속합니다')
            self.finish_source_transition(False, 'request failed')
            return
        self.source_advance_future = future
        self.source_advance_timeout_timer = self.create_timer(
            float(self.source_advance_timeout_seconds),
            self.source_advance_timed_out,
            clock=self.source_transition_clock)
        future.add_done_callback(self.source_advance_response)

    def source_advance_response(self, future):
        if future is not self.source_advance_future:
            self.get_logger().warning(
                '완료된 이전 가스원 전환 응답을 무시합니다')
            return
        self.source_advance_future = None
        self.cancel_source_advance_timeout()
        try:
            response = future.result()
        except Exception as error:  # keep patrol alive on service errors
            self.get_logger().warning(
                f'가스원 전환 서비스 실패({error!r}); '
                '기존 가스원으로 LRS를 계속합니다')
            self.finish_source_transition(False, 'service call failed')
            return
        if response is None or not response.success:
            message = ('empty response' if response is None else
                       str(response.message))
            self.get_logger().warning(
                f'가스원 전환 거부({message}); '
                '기존 가스원으로 LRS를 계속합니다')
            self.finish_source_transition(False, 'service rejected request')
            return
        message = str(response.message)
        if message.startswith('changed=true;'):
            source_changed = True
        elif message.startswith('changed=false;'):
            source_changed = False
        else:
            self.get_logger().warning(
                '가스원 전환 응답에 changed 상태가 없어 기존 가스원 '
                f'이벤트로 보수적으로 처리합니다: {message!r}')
            source_changed = False
        self.get_logger().info(
            f'가스원 전환 서비스 완료: changed={source_changed}; '
            '새 LRS 순찰을 시작합니다')
        self.finish_source_transition(
            source_changed, 'source service acknowledged')

    def source_advance_timed_out(self):
        future = self.source_advance_future
        if future is None:
            self.cancel_source_advance_timeout()
            return
        self.source_advance_future = None
        self.cancel_source_advance_timeout()
        future.cancel()
        self.get_logger().warning(
            f'가스원 전환 서비스가 '
            f'{float(self.source_advance_timeout_seconds):.1f}초 안에 '
            '응답하지 않아 기존 가스원으로 LRS를 '
            '계속합니다')
        self.finish_source_transition(False, 'service timeout')

    def cancel_source_advance_timeout(self):
        timer = self.source_advance_timeout_timer
        self.source_advance_timeout_timer = None
        if timer is None:
            return
        timer.cancel()
        self.destroy_timer(timer)

    def finish_source_transition(self, source_changed, outcome):
        self.cancel_source_advance_timeout()
        reason = self.source_transition_reason
        self.source_transition_reason = ''
        if source_changed:
            self.current_event_id = ''
        status = ('new source event' if source_changed else
                  'existing source event')
        self.start_lrs_lap(
            f'{reason}; source_transition={outcome}; using {status}')

    def return_to_lrs(self, reason):
        self.get_logger().info(f'HRS 종료 후 LRS 복귀: {reason}')
        self.commit_history_snapshot(
            f'hazard event ending after LRS lap {self.lrs_lap}')
        self.peak_returning = False
        self.active_hrs_route = None
        self.publish_empty_hrs_route()
        self.clear_hrs_candidates()

        self.start_lrs_lap(reason)

    def clear_hrs_candidates(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_hrs_candidates'
        marker.id = 0
        marker.action = Marker.DELETE
        self.hrs_candidates_pub.publish(marker)

    def complete_mapping(self, reason):
        self.commit_history_snapshot(
            f'mapping completion after LRS lap {self.lrs_lap}')
        self.publish_phase('COMPLETE')
        self.get_logger().info(
            f'=== Gas mapping 완료: {reason}; HRS cycles={self.hrs_cycles}, '
            f'measured_cells={len(self.sampled_variables)}, '
            f'unreachable={len(self.unreachable_variables)} ===')

    def finalize_hrs_gmrf_batch(self, reason):
        if self.hrs_gmrf_dirty:
            self.get_logger().warning(
                f'HRS 배치 종료 dirty GaBP 재시도: {reason}')
            if not self.update_gmrf(
                    f'{reason} dirty retry', compare_with_cg=False):
                return False
            self.hrs_gmrf_dirty = False
        self.compare_gmrf_with_cg(reason)
        return True

    def compare_gmrf_with_cg(self, reason):
        gabp_solution = self.gmrf.solution.copy()
        reference, info = self.gmrf.cg_reference(
            tolerance=float(self.gabp_tolerance))
        if reference is None:
            self.get_logger().warning(
                f'GaBP 비교용 CG가 수렴하지 않았습니다'
                f'({reason}, info={info})')
            return math.nan
        max_error = float(np.max(np.abs(gabp_solution - reference)))
        if max_error > float(self.gabp_cg_warning_tolerance):
            self.get_logger().warning(
                f'GaBP-CG 최대 평균 오차 {max_error:.3e}가 '
                f'허용값 {float(self.gabp_cg_warning_tolerance):.3e}를 '
                '초과했습니다')
        self.get_logger().info(
            f'GaBP-CG 배치 비교({reason}): '
            f'max_error={max_error:.3e}')
        return max_error

    def update_gmrf(self, reason, compare_with_cg=True):
        previous_solution = self.gmrf.solution.copy()
        previous_variance = self.gmrf.variance.copy()
        previous_message_precision = self.gmrf.message_precision.copy()
        previous_message_information = self.gmrf.message_information.copy()
        converged, iterations, residual = self.gmrf.solve_gabp(
            int(self.gabp_max_iterations), float(self.gabp_tolerance),
            float(self.gabp_damping))
        damping = float(self.gabp_damping)
        if not converged:
            damping = float(self.gabp_retry_damping)
            self.gmrf.reset_gabp_messages()
            converged, iterations, residual = self.gmrf.solve_gabp(
                int(self.gabp_max_iterations), float(self.gabp_tolerance),
                damping)
        if not converged:
            self.gmrf.solution = previous_solution
            self.gmrf.variance = previous_variance
            self.gmrf.message_precision = previous_message_precision
            self.gmrf.message_information = previous_message_information
            self.get_logger().warning(
                f'GaBP 미수렴({reason}, residual={residual:.3e}); '
                '마지막 정상 지도와 메시지를 유지합니다')
            return False
        if compare_with_cg:
            max_error = self.compare_gmrf_with_cg(reason)
            cg_text = f'{max_error:.3e}'
        else:
            cg_text = 'skipped'
        self.publish_maps()
        self.get_logger().info(
            f'GaBP 갱신({reason}): iterations={iterations}, '
            f'residual={residual:.3e}, damping={damping:.2f}, '
            f'CG_max_error={cg_text}, '
            f'variance=[{np.min(self.gmrf.variance):.3e},'
            f'{np.max(self.gmrf.variance):.3e}]')
        return True

    def display_field(self, variable_values):
        model_resolution = self.gmrf.resolution
        display_resolution = float(self.estimate_resolution)
        if display_resolution <= 0.0:
            raise ValueError('estimate_resolution must be positive')
        field = np.zeros((self.gmrf.height, self.gmrf.width), dtype=float)
        field[self.gmrf.free] = variable_values
        if math.isclose(display_resolution, model_resolution):
            return copy.deepcopy(self.map_msg), field, self.gmrf.free

        min_x = self.gmrf.origin_x + 0.5 * model_resolution
        min_y = self.gmrf.origin_y + 0.5 * model_resolution
        max_x = self.gmrf.origin_x + (
            self.gmrf.width - 0.5) * model_resolution
        max_y = self.gmrf.origin_y + (
            self.gmrf.height - 0.5) * model_resolution
        width = int(round((max_x - min_x) / display_resolution)) + 1
        height = int(round((max_y - min_y) / display_resolution)) + 1
        x_world = min_x + np.arange(width) * display_resolution
        y_world = min_y + np.arange(height) * display_resolution
        grid_x, grid_y = np.meshgrid(
            (x_world - min_x) / model_resolution,
            (y_world - min_y) / model_resolution)
        interpolated = map_coordinates(
            field, [grid_y, grid_x], order=1, mode='nearest')
        msg = copy.deepcopy(self.map_msg)
        msg.info.resolution = display_resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = min_x - 0.5 * display_resolution
        msg.info.origin.position.y = min_y - 0.5 * display_resolution
        return msg, interpolated, np.ones((height, width), dtype=bool)

    def occupancy_grid(self, template, values, free_mask):
        grid = copy.deepcopy(template)
        grid.header.stamp = self.get_clock().now().to_msg()
        data = np.full(values.shape, -1, dtype=np.int16)
        clipped = np.clip(values, 0.0, 1.0)
        data[free_mask] = np.rint(100.0 * clipped[free_mask]).astype(np.int16)
        grid.data = data.astype(np.int8).ravel().tolist()
        return grid

    def publish_maps(self):
        template, mean, free = self.display_field(self.gmrf.solution)
        self.estimate_pub.publish(self.occupancy_grid(template, mean, free))

        log_normalized = self.logarithmic_display(mean)
        self.estimate_log_pub.publish(
            self.occupancy_grid(template, log_normalized, free))

        variance_template, variance, variance_free = self.display_field(
            self.gmrf.variance)
        self.variance_pub.publish(self.occupancy_grid(
            variance_template, variance, variance_free))
        reward = normalized_ucb(
            self.gmrf.solution, self.gmrf.variance,
            float(self.hrs_ucb_k))
        reward_template, reward_values, reward_free = self.display_field(reward)
        self.reward_pub.publish(self.occupancy_grid(
            reward_template, reward_values, reward_free))

    def logarithmic_display(self, values):
        log_min = float(self.log_concentration_min)
        if log_min >= 0.0:
            raise ValueError('log_concentration_min must be negative')
        log_values = np.log(np.clip(values, math.exp(log_min), 1.0))
        return np.clip(
            (log_values - log_min) / -log_min, 0.0, 1.0)

    def publish_measurements(self):
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = self.get_clock().now().to_msg()
        poses.poses = list(self.measured_poses)
        self.poses_pub.publish(poses)
        self.values_pub.publish(Float32MultiArray(data=self.measured_values))

    def commit_history_snapshot(self, reason):
        if (not self.history_snapshot_pending or self.history is None or
                self.gmrf is None):
            return False
        try:
            count = self.history.accumulate_distribution(
                self.gmrf.solution, self.gmrf.var_cells)
        except ValueError as error:
            self.get_logger().error(
                f'GMRF 이력 지도 누적 실패({reason}): {error}')
            return False
        self.history_snapshot_pending = False
        self.publish_history()
        persisted = self.persist_history(reason)
        self.get_logger().info(
            f'GMRF 이력 지도 회차 {count} 누적: {reason}')
        return persisted

    def persist_history(self, reason):
        if self.history is None:
            return False
        try:
            self.history.save()
            self.get_logger().info(
                f'가스 이력 저장({reason}): {len(self.history.records)} cells, '
                f'distribution_count={self.history.history_count}, '
                f'{self.history.path}')
            return True
        except OSError as error:
            self.get_logger().error(f'가스 이력 저장 실패: {error}')
            return False

    def clear_history(self, request, response):
        del request
        if self.history is None:
            response.success = False
            response.message = 'GMRF/history map is not ready'
            return response
        try:
            self.history.clear()
        except OSError as error:
            response.success = False
            response.message = f'Failed to clear history: {error}'
            return response
        self.active_history_regions = []
        self.publish_history()
        response.success = True
        response.message = (
            'Gas history cleared; the active LRS route will be rebuilt '
            'at the next lap')
        self.get_logger().info(response.message)
        return response

    def publish_lrs_active_route(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for goal in self.lrs_goals + self.lrs_goals[:1]:
            pose = copy.deepcopy(goal)
            pose.header = path.header
            path.poses.append(pose)
        self.lrs_active_route_pub.publish(path)

    def publish_lrs_reward(self):
        if self.gmrf is None or self.lrs_reward_values is None:
            return
        template, values, free = self.display_field(self.lrs_reward_values)
        self.lrs_reward_pub.publish(
            self.occupancy_grid(template, values, free))

    def publish_lrs_priority_candidates(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_lrs_priority_candidates'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.48
        for point in self.lrs_priority_candidates:
            marker.points.append(Point(x=point.x, y=point.y, z=0.20))
            if point.mandatory:
                marker.colors.append(ColorRGBA(
                    r=0.0, g=1.0, b=0.55, a=1.0))
            else:
                marker.colors.append(ColorRGBA(
                    r=1.0, g=0.25 + 0.55 * point.reward,
                    b=0.0, a=1.0))
        self.lrs_priority_candidates_pub.publish(marker)

    def publish_lrs_priority_route(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        if self.latest_pose is not None and self.lrs_priority_points:
            start = copy.deepcopy(self.latest_pose)
            start.header = path.header
            path.poses.append(start)
        for point in self.lrs_priority_points:
            pose = self.make_pose(point.x, point.y)
            pose.header = path.header
            path.poses.append(pose)
        self.lrs_priority_route_pub.publish(path)

    def publish_history(self):
        if self.history is None or self.gmrf is None:
            return
        grid = copy.deepcopy(self.map_msg)
        grid.header.stamp = self.get_clock().now().to_msg()
        history_values = self.history.history_mean[self.gmrf.free]
        data = np.full(
            (self.gmrf.height, self.gmrf.width), -1, dtype=np.int16)
        data[self.gmrf.free] = np.rint(
            100.0 * np.clip(history_values, 0.0, 1.0)).astype(np.int16)
        grid.data = data.astype(np.int8).ravel().tolist()
        self.history_map_pub.publish(grid)

        log_template, interpolated, log_free = self.display_field(
            history_values)
        log_values = self.logarithmic_display(interpolated)
        self.history_log_map_pub.publish(self.occupancy_grid(
            log_template, log_values, log_free))

        recent_values = self.history.history_recent[self.gmrf.free]
        recent_template, recent_interpolated, recent_free = (
            self.display_field(recent_values))
        recent_log_values = self.logarithmic_display(recent_interpolated)
        self.history_recent_log_map_pub.publish(self.occupancy_grid(
            recent_template, recent_log_values, recent_free))

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_history_revisit'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.65
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = [
            Point(x=record['x'], y=record['y'], z=0.18)
            for record in self.active_history_regions]
        self.history_revisit_pub.publish(marker)

        peaks = Marker()
        peaks.header.frame_id = 'map'
        peaks.header.stamp = self.get_clock().now().to_msg()
        peaks.ns = 'gas_mapping_history_confirmed_peaks'
        peaks.id = 0
        peaks.type = Marker.POINTS
        peaks.action = Marker.ADD
        peaks.scale.x = peaks.scale.y = 0.72
        latest_sequence = self.history.confirmed_event_sequence
        events = sorted(
            self.history.confirmed_peak_events.values(),
            key=lambda event: (
                int(event['sequence']), str(event['event_id'])))
        for event in events:
            x, y = self.history.cell_center(event['row'], event['col'])
            peaks.points.append(Point(x=x, y=y, z=0.24))
            if int(event['sequence']) == latest_sequence:
                peaks.colors.append(ColorRGBA(
                    r=0.0, g=1.0, b=0.55, a=1.0))
            else:
                peaks.colors.append(ColorRGBA(
                    r=1.0, g=0.35, b=0.0, a=0.9))
        self.history_confirmed_peaks_pub.publish(peaks)

    def publish_empty_hrs_route(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        self.hrs_route_pub.publish(path)

    def value_color(self, value, logarithmic=False):
        if value is None:
            return ColorRGBA(r=0.55, g=0.55, b=0.55, a=1.0)
        display_value = min(1.0, max(0.0, float(value)))
        if logarithmic:
            log_min = float(self.log_concentration_min)
            clipped = max(math.exp(log_min), display_value)
            display_value = min(1.0, max(
                0.0, (math.log(clipped) - log_min) / -log_min))
        return ColorRGBA(
            r=display_value, g=0.0, b=1.0 - display_value, a=1.0)

    def publish_lrs_status(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_lrs_status'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.18
        marker.points = [
            Point(x=goal.pose.position.x, y=goal.pose.position.y, z=0.08)
            for goal in self.lrs_goals]
        marker.colors = [self.value_color(value)
                         for value in self.lrs_status_values]
        self.lrs_status_pub.publish(marker)
        log_marker = copy.deepcopy(marker)
        log_marker.ns = 'gas_mapping_lrs_status_log'
        log_marker.colors = [self.value_color(value, logarithmic=True)
                             for value in self.lrs_status_values]
        self.lrs_status_log_pub.publish(log_marker)

    def publish_candidates(self, candidates):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_hrs_candidates'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.35
        marker.points = [Point(x=cell.x, y=cell.y, z=0.14)
                         for cell in candidates]
        marker.colors = [ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
                         for _ in candidates]
        self.hrs_candidates_pub.publish(marker)

    def publish_hrs_route(self, plan):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        if self.latest_pose is not None:
            start = copy.deepcopy(self.latest_pose)
            start.header = path.header
            path.poses.append(start)
        for cell in plan.cells:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = cell.x
            pose.pose.position.y = cell.y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.hrs_route_pub.publish(path)

    def publish_hrs_status(self):
        if self.gmrf is None:
            return
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_hrs_status'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.12
        for variable in range(len(self.gmrf.var_cells)):
            x, y = self.gmrf.cell_center(variable)
            marker.points.append(Point(x=x, y=y, z=0.10))
            if variable in self.unreachable_variables:
                marker.colors.append(ColorRGBA(
                    r=0.0, g=0.0, b=0.0, a=1.0))
            else:
                marker.colors.append(self.value_color(
                    self.hrs_values.get(variable)))
        self.hrs_status_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GasMappingControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
