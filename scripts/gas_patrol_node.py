#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose, Wait
from std_msgs.msg import Float64
from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatus

PURIFY_ENABLED = False  # 정화 모드 정지 타이밍 버그로 인해 임시 비활성화

DETECT_THRESHOLD = 50.0
RESUME_THRESHOLD = 20.0
WAIT_TIMEOUT_SEC = 10
MONITOR_PERIOD_SEC = 0.5
MAX_PURIFY_ATTEMPTS = 3

# (-5.0, -4.0) 시작 -> 점점 좁혀지는 사각 소용돌이
WAYPOINTS = [
    (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0), (-5.0, -2.0),
    (3.0, -2.0), (3.0, 2.0), (-2.0, 1.0), (-2.0, 0.0), (0.0, 0.0)
]

# (-5.0, -4.0) 시작 -> 단순 왕복 경로
# WAYPOINTS = [(-5.0, -1.0), (4.0, -1.0)]

class GasPatrolNode(Node):
    def __init__(self):
        super().__init__('gas_patrol_node')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.wait_client = ActionClient(self, Wait, 'wait')

        self.create_subscription(
            Float64, '/gas_sensor/detected_concentration',
            self.concentration_callback, 10)

        self.state = 'PATROLLING'
        self.current_index = 0
        self.direction = 1  # +1: 바깥->중심, -1: 중심->바깥
        self.latest_concentration = 0.0

        self.nav_goal_handle = None
        self.wait_goal_handle = None
        self.monitor_timer = None
        self.purify_fail_count = 0
        self.suppress_detection = False

        self.get_logger().info(f'patrol route ({len(WAYPOINTS)} points): {WAYPOINTS}')
        self.send_next_waypoint()

    # ---------------- PATROLLING ----------------

    def send_next_waypoint(self):
        x, y = WAYPOINTS[self.current_index]
        self.get_logger().info(f'heading to waypoint[{self.current_index}] = ({x:.2f}, {y:.2f})')

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0

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
        if self.state != 'PATROLLING' or goal_handle is not self.nav_goal_handle:
            return  # PURIFYING으로 취소됐거나, 이미 선점된 stale goal의 결과는 무시
        
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f'NavigateToPose failed with status={status}, retrying same waypoint')
            self.send_next_waypoint()  # advance 없이 같은 목표 재시도
            return

        was_suppressed = self.suppress_detection
        self.suppress_detection = False  # 도착 완료 - 다시 감지 활성화

        if PURIFY_ENABLED and self.latest_concentration > DETECT_THRESHOLD and not was_suppressed:
            # 도착 시점에 이미 임계값을 넘었고, 감지 무시 모드가 아니면 바로 정화
            self.start_purifying()
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
        if (PURIFY_ENABLED and self.state == 'PATROLLING' and not self.suppress_detection
                and self.latest_concentration > DETECT_THRESHOLD):
            self.start_seeking()

    # ---------------- SEEKING ----------------
    def start_seeking(self):
        self.get_logger().info(
            f'=== PURIFYING start === concentration={self.latest_concentration:.1f} '
            f'(threshold={DETECT_THRESHOLD}, resume_threshold={RESUME_THRESHOLD}, timeout={WAIT_TIMEOUT_SEC}s)')
        self.state = 'SEEKING'

    # ---------------- PURIFYING ----------------

    def start_purifying(self):
        self.get_logger().info(
            f'=== PURIFYING start === concentration={self.latest_concentration:.1f} '
            f'(threshold={DETECT_THRESHOLD}, resume_threshold={RESUME_THRESHOLD}, timeout={WAIT_TIMEOUT_SEC}s)')
        self.state = 'PURIFYING'

        if self.nav_goal_handle is not None:
            cancel_future = self.nav_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(lambda f: self._begin_wait())
        else:
            self._begin_wait()

    def _begin_wait(self):
        wait_goal = Wait.Goal()
        wait_goal.time = Duration(sec=WAIT_TIMEOUT_SEC)

        self.wait_client.wait_for_server()
        future = self.wait_client.send_goal_async(wait_goal)
        future.add_done_callback(self.wait_goal_response_callback)

        self.monitor_timer = self.create_timer(MONITOR_PERIOD_SEC, self.monitor_during_purify)

    def wait_goal_response_callback(self, future):
        self.wait_goal_handle = future.result()
        if not self.wait_goal_handle.accepted:
            self.get_logger().warn('Wait goal rejected')
            self.finish_purifying()
            return
        result_future = self.wait_goal_handle.get_result_async()
        result_future.add_done_callback(self.wait_result_callback)

    def monitor_during_purify(self):
        self.get_logger().info(
            f'purifying... checking concentration={self.latest_concentration:.1f} '
            f'(resume below {RESUME_THRESHOLD})')
        if self.latest_concentration < RESUME_THRESHOLD and self.wait_goal_handle is not None:
            self.get_logger().info('concentration dropped below resume threshold - canceling wait early')
            self.wait_goal_handle.cancel_goal_async()

    def wait_result_callback(self, future):
        self.finish_purifying()

    def finish_purifying(self):
        if self.monitor_timer is not None:
            self.monitor_timer.cancel()
            self.monitor_timer = None
        self.state = 'PATROLLING'

        if self.latest_concentration < RESUME_THRESHOLD:
            # 정화 성공 - 같은 waypoint로 계속 진행
            self.get_logger().info(
                f'=== PURIFYING end (success) === concentration={self.latest_concentration:.1f} - resume patrol')
            self.purify_fail_count = 0
            self.send_next_waypoint()
        else:
            # 타임아웃까지 갔는데도 농도가 안 떨어짐 - 실패로 간주
            self.purify_fail_count += 1
            self.get_logger().warn(
                f'=== PURIFYING end (timeout) === concentration={self.latest_concentration:.1f} '
                f'- attempt {self.purify_fail_count}/{MAX_PURIFY_ATTEMPTS}')

            if self.purify_fail_count >= MAX_PURIFY_ATTEMPTS:
                self.get_logger().warn('max purify attempts reached - push through to waypoint ignoring gas')
                self.purify_fail_count = 0
                self.suppress_detection = True  # 도착할 때까지 감지 무시

            self.send_next_waypoint()  # advance 없이 같은 waypoint로 (suppress 상태면 끝까지 밀고감)


def main(args=None):
    rclpy.init(args=args)
    node = GasPatrolNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
