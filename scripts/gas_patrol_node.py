#!/usr/bin/env python3
import math
import random

import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory

from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Float64, Empty
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus
from gazebo_msgs.srv import SetEntityState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped

from nav_msgs.msg import OccupancyGrid
from kernel_dm_grid import KernelDMGrid


def _load_gas_model_sigma():
    pkg_dir = get_package_share_directory('icir_cleanroom')
    with open(pkg_dir + '/config/gas_model.yaml') as f:
        return yaml.safe_load(f)['gas_model']['sigma']


GAS_MODEL_SIGMA = _load_gas_model_sigma()

DETECT_THRESHOLD = 150.0
INITIAL_SOURCE_CONCENTRATION = 500.0
# 가스 소스 재배치 가능 범위 (외벽 ±6.55/±5.05에서 1m 마진)
GAS_SPAWN_X_MIN, GAS_SPAWN_X_MAX = -5.5, 5.5
GAS_SPAWN_Y_MIN, GAS_SPAWN_Y_MAX = -4.0, 4.0

# 장비 박스 (이름, x_min, x_max, y_min, y_max). worlds/cleanroom.world의
# equip_* 모델 pose/size와 반드시 일치시켜야 함. 로그에 장비내부/인접 여부를 남기는 용도.
EQUIPMENT_BOXES = [
    ('left_outer_1', -6.5, -4.0, -4.0, -2.0),
    ('left_outer_2', -6.5, -4.0, -1.0, 1.0),
    ('left_outer_3', -6.5, -4.0, 2.0, 4.0),
    ('left_inner_1', -3.0, -1.5, -4.0, -2.0),
    ('left_inner_2', -3.0, -1.5, 2.0, 4.0),
    ('right_outer_1', 4.0, 6.5, -4.0, -2.0),
    ('right_outer_2', 4.0, 6.5, -1.0, 1.0),
    ('right_outer_3', 4.0, 6.5, 2.0, 4.0),
    ('right_inner_1', 1.5, 3.0, -4.0, -2.0),
    ('right_inner_2', 1.5, 3.0, 2.0, 4.0),
]
EQUIPMENT_ADJACENT_MARGIN = 0.5


def classify_equipment_proximity(x, y):
    for name, x0, x1, y0, y1 in EQUIPMENT_BOXES:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return f'장비내부:{name}'
    for name, x0, x1, y0, y1 in EQUIPMENT_BOXES:
        m = EQUIPMENT_ADJACENT_MARGIN
        if (x0 - m) <= x <= (x1 + m) and (y0 - m) <= y <= (y1 + m):
            return f'장비인접:{name}'
    return '개방'
# SPIRAL (Ferri et al. 2009) 파라미터
SPIRAL_BASE_ARM = 0.3       # 나선 시작 팔 길이 (m)
SPIRAL_ARM_GROWTH = 0.2     # 두 팔마다 늘어나는 길이 (m)
SPIRAL_DIRECTIONS = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]  # E, N, W, S
MISS_LIMIT = 5              # 연속 MISS가 이 횟수 쌓이면 T_PI 완화
T_PI_RELAX = 20.0           # 완화 시 T_PI를 낮추는 양
SPIRAL_STEP_TIMEOUT_SEC = 5.0  # nav2가 이 시간 안에 끝내지 못하면 그 스텝은 MISS로 간주
# 도착(FOUND) 판정: ground truth 없이 SPIRAL 수렴으로만 판단.
# 나선을 이 팔 길이 이상으로 넓혀도 더 나은 지점(HIT)을 못 찾으면 지역 최댓값 = 소스 도착으로 간주.
SPIRAL_MAX_ARM = 1.5        # 나선 팔 길이 상한 (m)

# REFINING: SPIRAL이 상한에 도달해 지역 최댓값으로 판단한 뒤, 로봇이 서 있는 그 자리 대신
# Kernel DM 그리드의 argmax로 한 번 더 이동해서 FOUND 위치를 보정. 데이터 분석 결과 SPIRAL의
# 정지점은 그리드 추정치보다 부정확한 경우가 많았음(개방공간에서도 89%가 그리드가 더 정확).
REFINE_MIN_DISTANCE = 0.05     # argmax가 이미 이 거리 안이면 보정 이동 생략
REFINE_STEP_TIMEOUT_SEC = 5.0  # nav2가 이 시간 안에 끝내지 못하면 현재 위치에서 FOUND 확정

# PATROLLING 워치독: nav2가 이 시간 안에 waypoint에 도달 못하면 (끼임/응답없음)
# 해당 waypoint를 포기하고 다음으로 건너뜀
PATROL_STEP_TIMEOUT_SEC = 60.0
PATROL_MAX_RETRY = 2        # 같은 waypoint 실패 재시도 한계, 초과 시 다음으로 건너뜀

PURIFY_PERIOD_SEC = 0.1
GAS_RESPAWN_DELAY_SEC = 10.0
PURIFY_RATE = 10.0  # PURIFY_PERIOD_SEC마다 줄어드는 양

# (-5.0, -4.0) 시작 -> 점점 좁혀지는 사각 소용돌이
# WAYPOINTS = [
#     (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0), (-5.0, -2.0),
#     (3.0, -2.0), (3.0, 2.0), (-2.0, 1.0), (-2.0, 0.0), (0.0, 0.0)
# ]

# 클린룸 뱀(serpentine) 경로
# 로봇 시작: (0.0, -4.0) 중앙 복도 하단
# 중앙 복도 → 좌측 통로 하강 → 중앙 복도 → 우측 통로 상승 (S자 반복)
# 각 장비 y레벨(-3.0, 0.0, 3.0)과 통로 x(-3.5, 0.0, 3.5)를 경유해
# 모든 장비 전면을 근거리에서 커버
WAYPOINTS = [
    ( 0.0,  4.0),   # 중앙 복도 - 상단 장비 레벨
    (-3.5,  4.0),   # 좌측 통로 - 상단 장비
    (-3.5,  0.0),   # 좌측 통로 - 중앙 장비 (외벽 중간 장비)
    (-3.5, -4.0),   # 좌측 통로 - 하단 장비
    ( 0.0, -4.0),   # 중앙 복도 - 하단 장비 레벨
    ( 0.0,  0.0),   # 정중앙 (중앙 사각지대 커버)
    ( 0.0,  4.0),   # 중앙 복도 - 상단 (중앙 세로 복도 완주)
    ( 3.5, -4.0),   # 우측 통로 - 하단 장비
    ( 3.5,  0.0),   # 우측 통로 - 중앙 장비
    ( 3.5,  4.0),   # 우측 통로 - 상단 장비
]

class GasPatrolNode(Node):
    def __init__(self):
        super().__init__('gas_patrol_node')

        # 실행 버전 토글 (Ablation 실험용)
        #   use_grid_refine=False : SPIRAL만 사용 (나선 수렴 위치를 그대로 FOUND) - baseline
        #   use_grid_refine=True  : Kernel DM + SPIRAL 통합 (나선 수렴 후 그리드 argmax로 보정 이동)
        # 그리드(히트맵)는 두 버전 모두 계속 만들어 heatmap_est를 로그에 남긴다(추정오차 비교용).
        # 차이는 "로봇이 실제로 argmax로 이동하는지" 뿐이다.
        self.declare_parameter('use_grid_refine', True)
        self.use_grid_refine = self.get_parameter('use_grid_refine').value
        self.run_mode = 'SPIRAL+KernelDM' if self.use_grid_refine else 'SPIRAL-only'
        self.get_logger().info(f'=== RUN MODE: {self.run_mode} (use_grid_refine={self.use_grid_refine}) ===')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Lilienthal & Duckett 2004 (Kernel DM) 원 논문 파라미터 그대로 사용
        # sigma=0.15m(15cm), cutoff_radius=3*sigma=0.45m, w_min=10*센서개수(1개)
        # sigma는 물리 확산 모델(GAS_MODEL_SIGMA)과 매칭. cutoff_radius는 좁게(0.5m) 잡아서
        # 측정값이 먼 지점까지 과잉 확장되는 것을 최소화 - SPIRAL이 HIT할 때마다 소스
        # 근처에서 나선을 다시 좁혀 반복 측정하므로, peak 주변은 좁은 반경으로도 충분히
        # 채워지고 오히려 더 뾰족하고 정확한 추정이 나옴. 먼 곳의 미확인(구멍)은 상관없음.
        # w_min은 로봇 1대 + SPIRAL 희소 샘플링에 맞춰 논문값(10*센서개수)보다 낮게 조정
        self.grid = KernelDMGrid(
            origin_x=-6.53, origin_y=-5.05,
            resolution=0.05, width=262, height=202,
            sigma=GAS_MODEL_SIGMA, cutoff_radius=0.5, w_min=1.0
        )
        
        self.create_subscription(
            Float64, '/gas_sensor/detected_concentration',
            self.concentration_callback, 10)
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(
            Float64, '/gas_source/concentration', self.source_concentration_callback, 10)
        self.source_set_pub = self.create_publisher(Float64, '/gas_source/concentration_set', 10)
        self.attraction_reset_pub = self.create_publisher(Empty, '/gas_attraction_local_layer/reset', 10)
        self.set_entity_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.state_marker_pub = self.create_publisher(Marker, '/robot_state_marker', 10)
        self.heatmap_pub = self.create_publisher(OccupancyGrid, '/gas_source/estimated_heatmap', 1)
        # 히트맵 argmax = 로봇이 추정한 소스 위치 (ground truth /gas_source/pose와 비교하면 오차)
        self.estimated_pose_pub = self.create_publisher(PoseStamped, '/gas_source/estimated_pose', 1)
        

        self.state = 'PATROLLING'
        self.current_index = 0
        self.direction = 1  # +1: 바깥->중심, -1: 중심->바깥
        self.latest_concentration = 0.0
        self.current_x = 0.0
        self.current_y = 0.0

        self.nav_goal_handle = None
        self.spiral_dir_index = 0
        self.spiral_arm_count = 0
        self.spiral_arm_length = SPIRAL_BASE_ARM
        self.spiral_x = 0.0
        self.spiral_y = 0.0
        self.t_pi = DETECT_THRESHOLD
        self.consecutive_miss = 0
        self.spiral_step_id = 0
        self.spiral_timeout_timer = None

        self.refine_step_id = 0
        self.refine_timeout_timer = None

        self.patrol_step_id = 0
        self.patrol_timeout_timer = None
        self.patrol_retry_count = 0
        # 끼임/실패 누적 통계 (실행 전체 기간 누적, 로그로 확인)
        self.patrol_nav_fail_count = 0   # 상태1: nav2가 FAILED 응답한 누적 횟수
        self.patrol_timeout_count = 0    # 상태2: 무응답 타임아웃 누적 횟수

        self.source_concentration = None
        self.purify_timer = None
        self.respawn_timer = None

        self.get_logger().info(f'patrol route ({len(WAYPOINTS)} points): {WAYPOINTS}')
        self.send_next_waypoint()

        # 패트롤 시작 후 가스 소스를 랜덤 위치로 배치 (Gazebo 서비스 준비 대기 1초)
        self._initial_respawn_timer = self.create_timer(1.0, self._initial_respawn_once)

        # 0.5초 간격으로 heatmap 발행
        self.create_timer(0.5, self.publish_heatmap)
        # 1초 간격으로 추정 소스 위치(argmax) 발행 (데이터 기록/오차 분석용)
        self.create_timer(1.0, self.publish_estimated_pose)

    def _initial_respawn_once(self):
        self._initial_respawn_timer.cancel()
        self._initial_respawn_timer = None
        self.respawn_gas_source()

    def publish_heatmap(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.grid.resolution
        grid_msg.info.width = self.grid.width
        grid_msg.info.height = self.grid.height
        grid_msg.info.origin.position.x = self.grid.origin_x
        grid_msg.info.origin.position.y = self.grid.origin_y
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = self.grid.to_occupancy_grid_data()
        self.heatmap_pub.publish(grid_msg)

    def publish_estimated_pose(self):
        est = self.grid.get_argmax_position()
        if est is None:
            return  # 아직 유효한 추정이 없음(측정 부족)
        est_x, est_y = est
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = est_x
        pose_msg.pose.position.y = est_y
        pose_msg.pose.orientation.w = 1.0
        self.estimated_pose_pub.publish(pose_msg)

    STATE_COLORS = {
        'PATROLLING': (0.2, 0.8, 0.2),   # 초록
        'SEEKING':    (1.0, 0.6, 0.0),   # 주황
        'FOUND':      (1.0, 0.2, 0.2),   # 빨강
        'PURIFYING':  (0.2, 0.6, 1.0),   # 파랑
    }

    def _publish_state_marker(self):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'robot_state'
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = self.current_x
        m.pose.position.y = self.current_y
        m.pose.position.z = 0.8
        m.pose.orientation.w = 1.0
        m.scale.z = 0.3
        r, g, b = self.STATE_COLORS.get(self.state, (1.0, 1.0, 1.0))
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        m.text = self.state
        self.state_marker_pub.publish(m)

    def odom_callback(self, msg: Odometry):
        # map->odom이 identity static transform이라 odom 좌표를 map 좌표로 그대로 씀
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self._publish_state_marker()

    def source_concentration_callback(self, msg: Float64):
        self.source_concentration = msg.data

    def send_nav_goal(self, x, y):
        yaw = math.atan2(y - self.current_y, x - self.current_x)

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.nav_goal_response_callback)

    def nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('NavigateToPose goal rejected')
            return

        self.nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self.nav_result_callback(f, goal_handle))

    def nav_result_callback(self, future, goal_handle):
        if goal_handle is not self.nav_goal_handle:
            return  # 이미 선점된 stale goal의 결과는 무시

        status = future.result().status
        if status == GoalStatus.STATUS_CANCELED:
            return  # 의도적으로 취소한 경우 - 후속 동작은 취소를 호출한 쪽에서 직접 진행시킴

        if self.state == 'SEEKING':
            self.seek_step_result(status)
        elif self.state == 'REFINING':
            self.refine_step_result(status)
        else:
            self.patrol_step_result(status)

    # ---------------- PATROLLING ----------------

    def send_next_waypoint(self):
        # 방어: 어떤 이유로든 인덱스가 범위를 벗어나면 클램프 (노드 크래시 방지)
        if not (0 <= self.current_index < len(WAYPOINTS)):
            self.get_logger().warn(
                f'current_index({self.current_index}) 범위 벗어남 - 0으로 리셋')
            self.current_index = 0
            self.direction = 1
        x, y = WAYPOINTS[self.current_index]
        self.get_logger().info(f'heading to waypoint[{self.current_index}] = ({x:.2f}, {y:.2f})')
        self.send_nav_goal(x, y)

        # 워치독: 이 waypoint에 PATROL_STEP_TIMEOUT_SEC 안에 도달 못하면 건너뜀
        self.cancel_patrol_timeout()
        self.patrol_step_id += 1
        step_id = self.patrol_step_id
        self.patrol_timeout_timer = self.create_timer(
            PATROL_STEP_TIMEOUT_SEC, lambda: self.on_patrol_step_timeout(step_id))

    def cancel_patrol_timeout(self):
        if self.patrol_timeout_timer is not None:
            self.patrol_timeout_timer.cancel()
            self.patrol_timeout_timer = None

    # waypoint에 제한 시간 안에 도달하지 못하면 (끼임/nav2 무응답) 포기하고 다음 waypoint로
    def on_patrol_step_timeout(self, step_id):
        self.cancel_patrol_timeout()
        if self.state != 'PATROLLING' or step_id != self.patrol_step_id:
            return  # 이미 다음 스텝으로 넘어갔거나 PATROLLING을 벗어난 stale 타임아웃

        self.patrol_timeout_count += 1
        self.get_logger().warn(
            f'waypoint[{self.current_index}] 도달 실패 (timeout {PATROL_STEP_TIMEOUT_SEC}s) - 다음 waypoint로 건너뜀 '
            f'[상태2 무응답 타임아웃 누적 {self.patrol_timeout_count}회]')
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None
        self.patrol_retry_count = 0
        self.advance_waypoint()
        self.send_next_waypoint()

    def patrol_step_result(self, status):
        self.cancel_patrol_timeout()

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.patrol_nav_fail_count += 1
            self.patrol_retry_count += 1
            if self.patrol_retry_count <= PATROL_MAX_RETRY:
                self.get_logger().warn(
                    f'waypoint[{self.current_index}] 실패(status={status}) - 재시도 '
                    f'{self.patrol_retry_count}/{PATROL_MAX_RETRY} '
                    f'[상태1 nav2 실패 누적 {self.patrol_nav_fail_count}회]')
                self.send_next_waypoint()  # advance 없이 같은 목표 재시도
            else:
                self.get_logger().warn(
                    f'waypoint[{self.current_index}] 재시도 한계 초과 - 다음 waypoint로 건너뜀 '
                    f'[상태1 nav2 실패 누적 {self.patrol_nav_fail_count}회]')
                self.patrol_retry_count = 0
                self.advance_waypoint()
                self.send_next_waypoint()
            return

        self.patrol_retry_count = 0
        self.advance_waypoint()
        self.send_next_waypoint()

    def advance_waypoint(self):
        # 인덱스를 접근하기 전에 범위를 검사해 반전 (current_index가 redirect로
        # 임의 값이 되어도 out-of-range 크래시가 나지 않도록)
        next_index = self.current_index + self.direction
        if next_index >= len(WAYPOINTS):
            self.direction = -1
            next_index = self.current_index + self.direction
        elif next_index < 0:
            self.direction = 1
            next_index = self.current_index + self.direction
        self.current_index = next_index

    # ---------------- 감지 ----------------

    def concentration_callback(self, msg: Float64):
        self.latest_concentration = msg.data
        self.get_logger().info(
            f'concentration={self.latest_concentration:.1f}', throttle_duration_sec=1.0)
        self.grid.update(self.current_x, self.current_y, self.latest_concentration)

        if self.state == 'PATROLLING' and self.latest_concentration > DETECT_THRESHOLD:
            self.start_seeking()

    # ---------------- SEEKING (SPIRAL, Ferri et al. 2009) ----------------

    # PATROLLING -> SEEKING 전환: 현재 위치를 나선 중심으로 잡고 탐색 시작
    def start_seeking(self):
        self.get_logger().info(
            f'=== SEEKING start === concentration={self.latest_concentration:.1f} '
            f'(threshold={DETECT_THRESHOLD})')
        self.state = 'SEEKING'
        self.cancel_patrol_timeout()
        self.patrol_retry_count = 0
        self.t_pi = self.latest_concentration
        self.consecutive_miss = 0
        self.reset_spiral(self.current_x, self.current_y)

        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()

        self.begin_spiral_step()

    # 나선을 (cx, cy)를 시작점으로 초기화
    def reset_spiral(self, cx, cy):
        self.spiral_x = cx
        self.spiral_y = cy
        self.spiral_dir_index = 0
        self.spiral_arm_count = 0
        self.spiral_arm_length = SPIRAL_BASE_ARM

    # 현재 나선 위치에서 다음 방향으로 한 팔만큼 이동하는 골을 보내고 타임아웃 타이머 설정
    def begin_spiral_step(self):
        self.cancel_spiral_timeout()

        dx, dy = SPIRAL_DIRECTIONS[self.spiral_dir_index]
        target_x = self.spiral_x + dx * self.spiral_arm_length
        target_y = self.spiral_y + dy * self.spiral_arm_length
        self.get_logger().info(
            f'spiral step dir={self.spiral_dir_index} arm={self.spiral_arm_length:.2f} '
            f'-> ({target_x:.2f}, {target_y:.2f})')
        self.send_nav_goal(target_x, target_y)

        self.spiral_step_id += 1
        step_id = self.spiral_step_id
        self.spiral_timeout_timer = self.create_timer(
            SPIRAL_STEP_TIMEOUT_SEC, lambda: self.on_spiral_step_timeout(step_id))

    # 진행 중인 spiral 타임아웃 타이머가 있으면 취소
    def cancel_spiral_timeout(self):
        if self.spiral_timeout_timer is not None:
            self.spiral_timeout_timer.cancel()
            self.spiral_timeout_timer = None

    # 한 스텝이 SPIRAL_STEP_TIMEOUT_SEC 안에 끝나지 않으면 nav2 골을 취소하고 MISS 처리
    def on_spiral_step_timeout(self, step_id):
        self.cancel_spiral_timeout()
        if self.state != 'SEEKING' or step_id != self.spiral_step_id:
            return  # 이미 다음 스텝으로 넘어갔거나 SEEKING을 벗어난 stale 타임아웃

        self.get_logger().warn(
            f'spiral step dir={self.spiral_dir_index} timeout ({SPIRAL_STEP_TIMEOUT_SEC}s) - MISS로 간주')
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()

        self.spiral_step_failed()

    # 한 팔 이동 결과: 도착 지점의 측정농도(PI)를 T_PI(임계값)와 비교해 HIT/MISS 판단
    def seek_step_result(self, status):
        self.cancel_spiral_timeout()

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.spiral_step_failed()
            return

        proximity_index = self.latest_concentration

        if proximity_index >= self.t_pi:
            self.get_logger().info(
                f'spiral HIT (PI={proximity_index:.1f} >= T_PI={self.t_pi:.1f}) - 나선 좁혀서 재시작')
            self.t_pi = proximity_index
            self.consecutive_miss = 0
            self.reset_spiral(self.current_x, self.current_y)
            self.begin_spiral_step()
        else:
            self.spiral_step_failed()

    # MISS 처리: 다음 방향으로 전환, 두 팔마다 팔 길이 증가, 연속 MISS가 쌓이면 T_PI 완화
    def spiral_step_failed(self):
        self.consecutive_miss += 1
        self.spiral_dir_index = (self.spiral_dir_index + 1) % len(SPIRAL_DIRECTIONS)
        self.spiral_arm_count += 1
        self.spiral_arm_length = SPIRAL_BASE_ARM + (self.spiral_arm_count // 2) * SPIRAL_ARM_GROWTH

        # 나선을 상한까지 넓혔는데도 더 나은 지점(HIT)이 없으면 지역 최댓값 도달 = 소스 도착
        if self.spiral_arm_length > SPIRAL_MAX_ARM:
            if self.use_grid_refine:
                self.start_refine_to_argmax()   # 통합: 그리드 argmax로 보정 이동
            else:
                self.enter_found()              # SPIRAL만: 나선 수렴 위치를 그대로 확정
            return

        if self.consecutive_miss >= MISS_LIMIT:
            self.t_pi = max(0.0, self.t_pi - T_PI_RELAX)
            self.consecutive_miss = 0
            self.get_logger().info(f'연속 {MISS_LIMIT}회 MISS - T_PI를 {self.t_pi:.1f}로 완화')

        self.begin_spiral_step()

    # SPIRAL이 지역 최댓값에 도달한 뒤, 로봇이 서 있는 자리 대신 Kernel DM 그리드의
    # argmax로 한 번 더 이동해서 FOUND 위치를 보정 (SPIRAL 정지점보다 그리드 추정이 더 정확함)
    def start_refine_to_argmax(self):
        est = self.grid.get_argmax_position()
        if est is None or math.hypot(est[0] - self.current_x, est[1] - self.current_y) < REFINE_MIN_DISTANCE:
            self.enter_found()
            return

        self.get_logger().info(
            f'나선 상한 도달 - grid argmax로 최종 보정 이동: '
            f'({self.current_x:.2f}, {self.current_y:.2f}) -> ({est[0]:.2f}, {est[1]:.2f})')
        self.state = 'REFINING'
        self.cancel_spiral_timeout()
        self.send_nav_goal(est[0], est[1])

        self.refine_step_id += 1
        step_id = self.refine_step_id
        self.refine_timeout_timer = self.create_timer(
            REFINE_STEP_TIMEOUT_SEC, lambda: self.on_refine_timeout(step_id))

    def cancel_refine_timeout(self):
        if self.refine_timeout_timer is not None:
            self.refine_timeout_timer.cancel()
            self.refine_timeout_timer = None

    # 보정 이동이 REFINE_STEP_TIMEOUT_SEC 안에 끝나지 않으면 (장비 근처라 planner가 못 감)
    # nav2 골을 취소하고 도달 가능했던 위치에서 그대로 FOUND 확정
    def on_refine_timeout(self, step_id):
        self.cancel_refine_timeout()
        if self.state != 'REFINING' or step_id != self.refine_step_id:
            return  # 이미 다음 스텝으로 넘어갔거나 REFINING을 벗어난 stale 타임아웃

        self.get_logger().warn('grid argmax 보정 이동 timeout - 현재 위치에서 FOUND 확정')
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
        self.enter_found()

    # 보정 이동 결과와 무관하게(성공/실패 모두) 도달한 위치를 최종 FOUND 위치로 확정
    def refine_step_result(self, status):
        self.cancel_refine_timeout()
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('grid argmax 보정 이동 완료')
        else:
            self.get_logger().warn(
                f'grid argmax 보정 이동 실패(status={status}) - 도달 가능했던 위치에서 FOUND 확정')
        self.enter_found()

    # SEEKING 종료, 소스 위치에 도착했다고 판단하고 PURIFYING 단계로 진입
    def enter_found(self):
        est = self.grid.get_argmax_position()
        est_str = f'({est[0]:.2f}, {est[1]:.2f})' if est is not None else 'N/A'
        self.get_logger().info(
            f'=== FOUND === mode={self.run_mode} '
            f'robot_pos=({self.current_x:.2f}, {self.current_y:.2f}) '
            f'concentration={self.latest_concentration:.1f} heatmap_est={est_str} '
            f'- 나선 상한({SPIRAL_MAX_ARM}m) 도달, 소스 도착으로 판단')
        self.cancel_spiral_timeout()
        self.cancel_refine_timeout()
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
        self.state = 'FOUND'
        self.start_purifying()

    # ---------------- PURIFYING ----------------

    # PURIFYING 시작: 일정 주기(PURIFY_PERIOD_SEC)로 purify_step을 반복 호출하는 타이머 설정
    def start_purifying(self):
        self.get_logger().info('=== PURIFYING start === 가스 소스 강도를 서서히 낮춤')
        self.state = 'PURIFYING'
        self.purify_timer = self.create_timer(PURIFY_PERIOD_SEC, self.purify_step)

    # 거리 기반 근접도(proximity)에 비례해 가스 소스 농도를 줄여서 발행, 0이 되면 정화 완료 처리
    def purify_step(self):
        if self.source_concentration is None or self.source_concentration <= 0.0:
            self.get_logger().warn('source concentration 아직 수신 안됨 - purify step 대기')
            return

        # 거리를 직접 알 수는 없지만, 센서값/소스 출력값 비율이 곧 거리 감쇠율(gas_sensor_plugin의
        # (1/distance^2)과 같으므로 이를 "가까운 정도"로 써서 정화 속도에 반영
        proximity = min(1.0, self.latest_concentration / self.source_concentration)
        decrement = PURIFY_RATE * proximity

        new_value = max(0.0, self.source_concentration - decrement)
        self.source_set_pub.publish(Float64(data=new_value))
        self.get_logger().info(
            f'purifying... proximity={proximity:.2f} source concentration '
            f'{self.source_concentration:.1f} -> {new_value:.1f}')

        if new_value <= 0.0:
            self.get_logger().info('=== PURIFYING complete === 가스 소스 제거 완료, 5초 후 랜덤 위치로 재생성')
            self.attraction_reset_pub.publish(Empty())
            self.grid.reset()  # 정화 완료 시 추정 히트맵 초기화
            self.resume_patrolling()
            self.respawn_timer = self.create_timer(GAS_RESPAWN_DELAY_SEC, self._respawn_once)

    # 5초 지연 원샷 콜백 - 한 번 실행 후 타이머 자체를 취소
    def _respawn_once(self):
        if self.respawn_timer is not None:
            self.respawn_timer.cancel()
            self.respawn_timer = None
        self.respawn_gas_source()

    # 가스 소스를 맵 내 랜덤 위치로 이동하고 초기 농도로 재설정, 가장 가까운 waypoint로 우선 이동
    def respawn_gas_source(self):
        # 새 에피소드 시작: 진행 중이던 탐색/정화를 정리하고 PATROLLING으로 리셋
        self.cancel_spiral_timeout()
        self.cancel_refine_timeout()
        self.cancel_patrol_timeout()
        self.patrol_retry_count = 0
        if self.purify_timer is not None:
            self.purify_timer.cancel()
            self.purify_timer = None
        self.state = 'PATROLLING'

        x = random.uniform(GAS_SPAWN_X_MIN, GAS_SPAWN_X_MAX)
        y = random.uniform(GAS_SPAWN_Y_MIN, GAS_SPAWN_Y_MAX)
        proximity = classify_equipment_proximity(x, y)
        self.get_logger().info(
            f'=== GAS SOURCE RESPAWN === 새 위치: ({x:.2f}, {y:.2f}) [{proximity}]')
        self.get_logger().info(
            f'[누적 통계] 상태1 nav2 실패 {self.patrol_nav_fail_count}회, '
            f'상태2 무응답 타임아웃 {self.patrol_timeout_count}회')
        self._redirect_patrol_to_nearest(x, y)

        req = SetEntityState.Request()
        req.state.name = 'gas_source'
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.5
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = 'world'

        if self.set_entity_state_client.service_is_ready():
            future = self.set_entity_state_client.call_async(req)
            future.add_done_callback(
                lambda f: self.get_logger().info('가스 소스 위치 이동 완료'))
        else:
            self.get_logger().warn('/gazebo/set_entity_state 서비스 미준비 - 위치 이동 건너뜀')

        # 가스 소스 농도를 초기값으로 재설정
        self.source_set_pub.publish(Float64(data=INITIAL_SOURCE_CONCENTRATION))
        self.source_concentration = INITIAL_SOURCE_CONCENTRATION

    # 가스 발생 위치(x, y)에서 가장 가까운 waypoint로 patrol 경로 재설정
    def _redirect_patrol_to_nearest(self, x, y):
        nearest_idx = min(
            range(len(WAYPOINTS)),
            key=lambda i: (WAYPOINTS[i][0] - x) ** 2 + (WAYPOINTS[i][1] - y) ** 2
        )
        self.current_index = nearest_idx
        self.get_logger().info(
            f'=== GAS ALERT === waypoint[{nearest_idx}] {WAYPOINTS[nearest_idx]} 으로 우선 이동')

        if self.state == 'PATROLLING':
            if self.nav_goal_handle is not None:
                self.nav_goal_handle.cancel_goal_async()
                self.nav_goal_handle = None  # stale cancel 결과 무시
            self.send_next_waypoint()

    # PURIFYING/FOUND 종료, 정화 타이머와 진행 중인 골 정리 후 PATROLLING으로 복귀
    def resume_patrolling(self):
        self.get_logger().info(
            f'=== {self.state} end === concentration={self.latest_concentration:.1f} - resume patrol')
        self.state = 'PATROLLING'

        if self.purify_timer is not None:
            self.purify_timer.cancel()
            self.purify_timer = None

        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()

        self.send_next_waypoint()  # advance 없이 같은 waypoint로 재출발


def main(args=None):
    rclpy.init(args=args)
    node = GasPatrolNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
