#!/usr/bin/env python3
import math
import random

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Float64, Empty
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus
from gazebo_msgs.srv import SetEntityState

DETECT_THRESHOLD = 100.0
INITIAL_SOURCE_CONCENTRATION = 500.0
# 가스 소스 재배치 가능 범위 (외벽 ±6.55/±5.05에서 1m 마진)
GAS_SPAWN_X_MIN, GAS_SPAWN_X_MAX = -5.5, 5.5
GAS_SPAWN_Y_MIN, GAS_SPAWN_Y_MAX = -4.0, 4.0
SEEK_STEP = 0.4
SEEK_DIRECTIONS = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]  # E, N, W, S
IMPROVEMENT_MARGIN = 5.0  # 이 이상 커져야 "개선"으로 인정
SEEK_STEP_TIMEOUT_SEC = 3.0  # nav2가 이 시간 안에 끝내지 못하면 그 방향은 실패로 간주

PURIFY_PERIOD_SEC = 0.1
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
    ( 0.0,  3.0),   # 중앙 복도 - 상단 장비 레벨
    (-3.5,  3.0),   # 좌측 통로 - 상단 장비
    (-3.5,  0.0),   # 좌측 통로 - 중앙 장비 (외벽 중간 장비)
    (-3.5, -3.0),   # 좌측 통로 - 하단 장비
    ( 0.0, -3.0),   # 중앙 복도 - 하단 장비 레벨
    ( 3.5, -3.0),   # 우측 통로 - 하단 장비
    ( 3.5,  0.0),   # 우측 통로 - 중앙 장비
    ( 3.5,  3.0),   # 우측 통로 - 상단 장비
]

class GasPatrolNode(Node):
    def __init__(self):
        super().__init__('gas_patrol_node')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

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

        self.state = 'PATROLLING'
        self.current_index = 0
        self.direction = 1  # +1: 바깥->중심, -1: 중심->바깥
        self.latest_concentration = 0.0
        self.current_x = 0.0
        self.current_y = 0.0

        self.nav_goal_handle = None
        self.seek_dir_index = 0
        self.prev_seek_concentration = 0.0
        self.no_improvement_count = 0
        self.seek_step_id = 0
        self.seek_timeout_timer = None

        self.source_concentration = None
        self.purify_timer = None
        self.respawn_timer = None

        self.get_logger().info(f'patrol route ({len(WAYPOINTS)} points): {WAYPOINTS}')
        self.send_next_waypoint()

    def odom_callback(self, msg: Odometry):
        # map->odom이 identity static transform이라 odom 좌표를 map 좌표로 그대로 씀
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

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
        else:
            self.patrol_step_result(status)

    # ---------------- PATROLLING ----------------

    def send_next_waypoint(self):
        x, y = WAYPOINTS[self.current_index]
        self.get_logger().info(f'heading to waypoint[{self.current_index}] = ({x:.2f}, {y:.2f})')
        self.send_nav_goal(x, y)

    def patrol_step_result(self, status):
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f'NavigateToPose failed with status={status}, retrying same waypoint')
            self.send_next_waypoint()  # advance 없이 같은 목표 재시도
            return

        self.advance_waypoint()
        self.send_next_waypoint()

    def advance_waypoint(self):
        self.current_index += self.direction
        if self.current_index == len(WAYPOINTS) - 1:
            self.direction = -1   # 중심 도달 -> 반대 방향으로 전환
        elif self.current_index == 0:
            self.direction = 1    # 바깥 끝 도달 -> 정방향으로 전환

    # ---------------- 감지 ----------------

    def concentration_callback(self, msg: Float64):
        self.latest_concentration = msg.data
        self.get_logger().info(
            f'concentration={self.latest_concentration:.1f}', throttle_duration_sec=1.0)

        if self.state == 'PATROLLING' and self.latest_concentration > DETECT_THRESHOLD:
            self.start_seeking()
        elif self.state in ('SEEKING', 'FOUND') and self.latest_concentration <= DETECT_THRESHOLD:
            self.resume_patrolling()

    # ---------------- SEEKING (gradient ascent) ----------------

    # PATROLLING -> SEEKING 전환: 현재 진행 중인 patrol 골을 취소하고 탐색 시작
    def start_seeking(self):
        self.get_logger().info(
            f'=== SEEKING start === concentration={self.latest_concentration:.1f} '
            f'(threshold={DETECT_THRESHOLD})')
        self.state = 'SEEKING'
        self.seek_dir_index = 0
        self.no_improvement_count = 0

        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()

        self.begin_seek_step()

    # 현재 방향(seek_dir_index)으로 한 스텝(SEEK_STEP) 이동하는 골을 보내고 타임아웃 타이머 설정
    def begin_seek_step(self):
        self.cancel_seek_timeout()

        self.prev_seek_concentration = self.latest_concentration
        dx, dy = SEEK_DIRECTIONS[self.seek_dir_index]
        target_x = self.current_x + dx * SEEK_STEP
        target_y = self.current_y + dy * SEEK_STEP
        self.get_logger().info(
            f'seek step dir={self.seek_dir_index} -> ({target_x:.2f}, {target_y:.2f})')
        self.send_nav_goal(target_x, target_y)

        self.seek_step_id += 1
        step_id = self.seek_step_id
        self.seek_timeout_timer = self.create_timer(
            SEEK_STEP_TIMEOUT_SEC, lambda: self.on_seek_step_timeout(step_id))

    # 진행 중인 seek 타임아웃 타이머가 있으면 취소
    def cancel_seek_timeout(self):
        if self.seek_timeout_timer is not None:
            self.seek_timeout_timer.cancel()
            self.seek_timeout_timer = None

    # 한 스텝이 SEEK_STEP_TIMEOUT_SEC 안에 끝나지 않으면 nav2 골을 취소하고 실패 처리
    def on_seek_step_timeout(self, step_id):
        self.cancel_seek_timeout()
        if self.state != 'SEEKING' or step_id != self.seek_step_id:
            return  # 이미 다음 스텝으로 넘어갔거나 SEEKING을 벗어난 stale 타임아웃

        self.get_logger().warn(
            f'seek step dir={self.seek_dir_index} timeout ({SEEK_STEP_TIMEOUT_SEC}s) - 실패로 간주')
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()

        self.seek_step_failed()

    # 스텝 이동 결과(nav2 골 status)를 보고 농도가 충분히 올랐으면 같은 방향 계속, 아니면 실패 처리
    def seek_step_result(self, status):
        self.cancel_seek_timeout()

        improved = status == GoalStatus.STATUS_SUCCEEDED and \
            self.latest_concentration - self.prev_seek_concentration > IMPROVEMENT_MARGIN

        if improved:
            self.no_improvement_count = 0
            self.begin_seek_step()
        else:
            self.seek_step_failed()

    # 현재 방향 실패 처리 후 다음 방향으로 전환, 모든 방향이 다 실패하면 FOUND로 전환
    def seek_step_failed(self):
        self.no_improvement_count += 1
        self.seek_dir_index = (self.seek_dir_index + 1) % len(SEEK_DIRECTIONS)

        if self.no_improvement_count >= len(SEEK_DIRECTIONS):
            self.enter_found()
        else:
            self.begin_seek_step()

    # SEEKING 종료, 소스 위치를 찾았다고 판단하고 PURIFYING 단계로 진입
    def enter_found(self):
        self.get_logger().info(
            f'=== FOUND === concentration={self.latest_concentration:.1f} '
            f'- {len(SEEK_DIRECTIONS)}방향 모두 개선 없음, 탐색 종료하고 정지')
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
            self.resume_patrolling()
            self.respawn_timer = self.create_timer(5.0, self._respawn_once)

    # 5초 지연 원샷 콜백 - 한 번 실행 후 타이머 자체를 취소
    def _respawn_once(self):
        if self.respawn_timer is not None:
            self.respawn_timer.cancel()
            self.respawn_timer = None
        self.respawn_gas_source()

    # 가스 소스를 맵 내 랜덤 위치로 이동하고 초기 농도로 재설정
    def respawn_gas_source(self):
        x = random.uniform(GAS_SPAWN_X_MIN, GAS_SPAWN_X_MAX)
        y = random.uniform(GAS_SPAWN_Y_MIN, GAS_SPAWN_Y_MAX)
        self.get_logger().info(f'=== GAS SOURCE RESPAWN === 새 위치: ({x:.2f}, {y:.2f})')

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
