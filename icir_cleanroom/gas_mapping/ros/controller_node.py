"""ROS adapter for continuous LRS/HRS gas mapping."""

import copy
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import Trigger

from ..application.hrs import HrsManager
from ..application.lrs import LrsManager
from ..application.measurement import MeasurementManager
from ..application.navigation import NavigationManager
from ..application.orchestrator import MappingOrchestrator
from ..application.peak_confirmation import PeakConfirmationManager
from ..application.planning_executor import PlanningExecutor
from ..config import declare_controller_config
from ..history import GasHistoryStore
from ..mapping.gmrf import GmrfGrid
from ..mapping.gmrf_service import GmrfService
from ..mapping.grid_geometry import GridGeometry
from ..mapping.path_distance import SamplingDistanceOracle
from ..models import (
    HrsRuntimeState, LrsRuntimeState, MappingEventState, NavigationState,
    PeakSearchState, SourceTransitionState, StateField)
from ..phase_machine import PhaseMachine
from .nav2_client import Nav2Client
from .hrs_workflow import HrsWorkflow
from .lrs_workflow import LrsWorkflow
from .navigation_workflow import NavigationWorkflow
from .peak_workflow import PeakWorkflow
from .planning_workflow import PlanningWorkflow
from .publishers import ControllerVisualization, MappingPublishers
from .source_transition import SourceTransitionWorkflow


def transient_qos():
    qos = QoSProfile(depth=1)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


class GasMappingControllerNode(Node):
    base_lrs_goals = StateField('lrs_state', 'base_goals')
    lrs_goals = StateField('lrs_state', 'goals')
    lrs_status_values = StateField('lrs_state', 'status_values')
    active_history_regions = StateField(
        'lrs_state', 'active_history_regions')
    lrs_reward_values = StateField('lrs_state', 'reward_values')
    lrs_reward_points = StateField('lrs_state', 'reward_points')
    lrs_priority_candidates = StateField(
        'lrs_state', 'priority_candidates')
    lrs_priority_points = StateField('lrs_state', 'priority_points')
    lrs_lap = StateField('lrs_state', 'lap')
    lap_max_concentration = StateField(
        'lrs_state', 'lap_max_concentration')
    lap_hazard_detected = StateField('lrs_state', 'hazard_detected')

    current_index = StateField('navigation_state', 'index')
    retry = StateField('navigation_state', 'retry_count')
    returning = StateField('navigation_state', 'returning')

    sampled_variables = StateField('event_state', 'sampled_variables')
    hrs_values = StateField('event_state', 'measured_by_variable')
    event_best_observed = StateField('event_state', 'best_value')
    event_best_variable = StateField('event_state', 'best_variable')
    history_snapshot_pending = StateField(
        'event_state', 'snapshot_pending')
    current_event_id = StateField('event_state', 'event_id')

    active_hrs_route = StateField('hrs_state', 'active_route')
    hrs_cycles = StateField('hrs_state', 'cycles')
    hrs_batch_successes = StateField('hrs_state', 'batch_successes')
    hrs_cycle_started_ns = StateField('hrs_state', 'cycle_started_ns')
    hrs_cycles_in_alert = StateField('hrs_state', 'cycles_in_alert')
    failure_counts = StateField('hrs_state', 'failure_counts')
    unreachable_variables = StateField(
        'hrs_state', 'unreachable_variables')
    hrs_gmrf_dirty = StateField('hrs_state', 'dirty')

    peak_anchor_variable = StateField('peak_state', 'anchor_variable')
    peak_move_count = StateField('peak_state', 'move_count')
    peak_trigger_reason = StateField('peak_state', 'trigger_reason')
    peak_returning = StateField('peak_state', 'returning')
    source_transition_reason = StateField(
        'source_transition_state', 'reason')

    def __init__(self):
        super().__init__('gas_mapping_controller_node')
        self.config = declare_controller_config(self)
        # Transitional aliases keep the parameter API stable while workflow
        # services adopt their immutable config groups.
        for name, value in self.config.flat_values().items():
            setattr(self, name, value)
        self.navigation_state = NavigationState()
        self.lrs_state = LrsRuntimeState()
        self.event_state = MappingEventState()
        self.hrs_state = HrsRuntimeState()
        self.peak_state = PeakSearchState()
        self.source_transition_state = SourceTransitionState()
        self.gmrf_service = GmrfService(self.config.gmrf)

        self.create_subscription(
            Path, '/gas_mapping/lrs/route', self.path_callback,
            transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_mapping/gmrf_domain', self.map_callback,
            transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_mapping/sampling_domain',
            self.sampling_map_callback, transient_qos())
        self.create_subscription(
            OccupancyGrid, '/gas_mapping/navigation_goal_domain',
            self.navigation_goal_map_callback, transient_qos())
        self.create_subscription(
            PoseStamped, '/gas_sensor/sensor_pose', self.pose_callback, 30)
        self.create_subscription(
            Float64, '/gas_mapping/sensor_concentration',
            self.value_callback, 30)

        self.mapping_publishers = MappingPublishers(self, transient_qos())
        # Method migration keeps these aliases temporarily; publisher
        # ownership itself is centralized in MappingPublishers.
        for name, publisher in self.mapping_publishers.handles.items():
            setattr(self, name, publisher)
        self.visualization = ControllerVisualization(self)
        for name in ControllerVisualization.METHODS:
            setattr(self, name, getattr(self.visualization, name))
        self.create_service(
            Trigger, '/gas_mapping/history/clear', self.clear_history)
        self.source_advance_client = self.create_client(
            Trigger, '/gas_mapping/source/advance')

        self.map_msg = None
        self.sampling_map_msg = None
        self.sampling_geometry = None
        self.sampling_mask = None
        self.sampling_distance = None
        self.navigation_goal_map_msg = None
        self.navigation_goal_geometry = None
        self.navigation_goal_mask = None
        self.navigation_goal_variables = None
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
        self.dwell_timer = None
        self.sampled_variables = set()
        self.hrs_values = {}
        self.failure_counts = {}
        self.unreachable_variables = set()

        self.phase_machine = PhaseMachine()
        self.phase = self.phase_machine.phase
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
        self.navigation_manager = NavigationManager(self.navigation_state)
        self.lrs_manager = LrsManager(self.lrs_state)
        self.hrs_manager = HrsManager(self.hrs_state)
        self.peak_manager = PeakConfirmationManager(self.peak_state)
        self.measurement_manager = MeasurementManager()
        self.orchestrator = MappingOrchestrator(
            self.phase_machine, self.navigation_manager, self.lrs_manager,
            self.hrs_manager, self.peak_manager, self.event_state,
            self.source_transition_state)
        self.planning_executor = PlanningExecutor()
        self.nav2 = Nav2Client(self, self.navigation_manager)
        self.workflow_adapters = [
            LrsWorkflow(self), HrsWorkflow(self), PlanningWorkflow(self),
            PeakWorkflow(self), NavigationWorkflow(self),
            SourceTransitionWorkflow(self),
        ]
        for workflow in self.workflow_adapters:
            for name in workflow.METHODS:
                setattr(self, name, getattr(workflow, name))
        self.start_timer = self.create_timer(1.0, self.try_start)
        self.planning_timer = self.create_timer(0.2, self.poll_planning)
        self.publish_phase('WAITING')
        self.hazard_pub.publish(Bool(data=False))
        self.get_logger().info(
            'LRS route, GMRF domain, sensor pose와 Nav2를 기다리는 중...')

    def destroy_node(self):
        self.planning_executor.shutdown()
        self.measurement_manager.cancel()
        self.navigation_manager.cancel_active_goal()
        if self.dwell_timer is not None:
            self.dwell_timer.cancel()
            self.destroy_timer(self.dwell_timer)
            self.dwell_timer = None
        self.cancel_source_advance_timeout()
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
        self.gmrf_service.attach(self.gmrf)
        self.refresh_navigation_goal_variables()
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
            f'GMRF 생성: {len(self.gmrf.solution)} field variables')

    def sampling_map_callback(self, msg):
        self.sampling_map_msg = copy.deepcopy(msg)
        self.sampling_geometry = GridGeometry.from_message(msg)
        values = np.asarray(msg.data, dtype=np.int16)
        expected = self.sampling_geometry.width * self.sampling_geometry.height
        if values.size != expected:
            self.get_logger().error(
                'Sampling domain data size does not match its geometry')
            self.sampling_distance = None
            return
        self.sampling_mask = values.reshape(
            self.sampling_geometry.height,
            self.sampling_geometry.width) == 0
        self.sampling_distance = SamplingDistanceOracle(
            self.sampling_geometry, self.sampling_mask)
        if (self.navigation_goal_geometry is not None and
                self.navigation_goal_geometry != self.sampling_geometry):
            self.get_logger().error(
                'Navigation goal and sampling domain geometries differ')
            self.navigation_goal_variables = None

    def navigation_goal_map_callback(self, msg):
        self.navigation_goal_map_msg = copy.deepcopy(msg)
        self.navigation_goal_geometry = GridGeometry.from_message(msg)
        values = np.asarray(msg.data, dtype=np.int16)
        expected = (
            self.navigation_goal_geometry.width *
            self.navigation_goal_geometry.height)
        if values.size != expected:
            self.get_logger().error(
                'Navigation goal domain data size does not match its '
                'geometry')
            self.navigation_goal_variables = None
            return
        if (self.sampling_geometry is not None and
                self.navigation_goal_geometry != self.sampling_geometry):
            self.get_logger().error(
                'Navigation goal and sampling domain geometries differ')
            self.navigation_goal_variables = None
            return
        self.navigation_goal_mask = values.reshape(
            self.navigation_goal_geometry.height,
            self.navigation_goal_geometry.width) == 0
        self.refresh_navigation_goal_variables()
        self.publish_hrs_status()

    def is_sampling_position(self, x, y):
        if self.sampling_geometry is None or self.sampling_mask is None:
            return False
        row, col = self.sampling_geometry.world_to_cell(x, y)
        return bool(
            self.sampling_geometry.contains_cell(row, col) and
            self.sampling_mask[row, col])

    def is_navigation_goal_position(self, x, y):
        if (self.navigation_goal_geometry is None or
                self.navigation_goal_mask is None):
            return False
        row, col = self.navigation_goal_geometry.world_to_cell(x, y)
        return bool(
            self.navigation_goal_geometry.contains_cell(row, col) and
            self.navigation_goal_mask[row, col])

    def refresh_navigation_goal_variables(self):
        if self.gmrf is None or self.navigation_goal_mask is None:
            return
        self.navigation_goal_variables = {
            variable for variable in range(len(self.gmrf.var_cells))
            if self.is_navigation_goal_position(
                *self.gmrf.cell_center(variable))}
        self.get_logger().info(
            f'Navigation goal domain aligned to GMRF: '
            f'{len(self.navigation_goal_variables)}/'
            f'{len(self.gmrf.var_cells)} cells')

    def pose_callback(self, msg):
        self.latest_pose = copy.deepcopy(msg)

    def value_callback(self, msg):
        self.measurement_manager.add_sample(msg.data)

    def try_start(self):
        if (self.phase != 'WAITING' or not self.base_lrs_goals or
                self.gmrf is None or self.latest_pose is None or
                self.sampling_distance is None or
                self.navigation_goal_variables is None or
                not self.nav2.server_is_ready()):
            return
        self.start_timer.cancel()
        self.start_lrs_lap('initial patrol')





    @staticmethod
    def make_pose(x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        return pose

    def publish_phase(self, phase):
        self.phase = self.phase_machine.transition(phase)
        phase_text = str(self.phase)
        self.phase_pub.publish(String(data=phase_text))
        self.get_logger().info(f'Gas mapping phase: {phase_text}')

    @staticmethod
    def is_hrs_navigation_phase(phase):
        return phase in ('HRS_NAVIGATION', 'HRS_PEAK_CONFIRMATION')

    def record_event_measurement(self, variable, value):
        self.orchestrator.record_event_measurement(
            variable, value, self.gmrf.var_cells)
































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
        max_error, info = self.gmrf_service.compare_with_cg()
        if not math.isfinite(max_error):
            self.get_logger().warning(
                f'GaBP 비교용 CG가 수렴하지 않았습니다'
                f'({reason}, info={info})')
            return math.nan
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
        result = self.gmrf_service.update(compare_with_cg=False)
        if not result.success:
            self.get_logger().warning(
                f'GaBP 미수렴({reason}, residual={result.residual:.3e}); '
                '마지막 정상 지도와 메시지를 유지합니다')
            return False
        if compare_with_cg:
            max_error = self.compare_gmrf_with_cg(reason)
            cg_text = f'{max_error:.3e}'
        else:
            cg_text = 'skipped'
        self.publish_maps()
        self.get_logger().info(
            f'GaBP 갱신({reason}): iterations={result.iterations}, '
            f'residual={result.residual:.3e}, damping={result.damping:.2f}, '
            f'CG_max_error={cg_text}, '
            f'variance=[{np.min(self.gmrf.variance):.3e},'
            f'{np.max(self.gmrf.variance):.3e}]')
        return True






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
