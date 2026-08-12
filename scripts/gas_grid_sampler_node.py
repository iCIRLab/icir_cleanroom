#!/usr/bin/env python3
import math

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class GasGridSamplerNode(Node):
    """Visit the DFS sampling points once each and dwell at every point."""

    def __init__(self):
        super().__init__('gas_grid_sampler_node')

        self.declare_parameter('dwell_seconds', 5.0)
        self.declare_parameter('max_retries', 2)
        self.dwell_seconds = float(
            self.get_parameter('dwell_seconds').value
        )
        self.max_retries = int(self.get_parameter('max_retries').value)

        path_qos = QoSProfile(depth=1)
        path_qos.reliability = ReliabilityPolicy.RELIABLE
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Path, '/gas_grid/dfs_path', self.path_callback, path_qos
        )

        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )
        self.sampling_poses = []
        self.current_index = 0
        self.retry_count = 0
        self.goal_in_progress = False
        self.route_started = False
        self.dwell_timer = None
        self.server_timer = self.create_timer(1.0, self.try_start_route)

        self.get_logger().info(
            'DFS 샘플링 경로와 Nav2 서버를 기다리는 중...'
        )

    def path_callback(self, msg: Path):
        if self.route_started:
            return

        # DFS 백트래킹으로 재방문하는 셀은 다시 샘플링하지 않는다.
        unique_poses = []
        visited = set()
        for pose in msg.poses:
            key = (
                round(pose.pose.position.x, 6),
                round(pose.pose.position.y, 6),
            )
            if key in visited:
                continue
            visited.add(key)
            unique_poses.append(pose)

        self.sampling_poses = unique_poses
        self.get_logger().info(
            f'샘플링 경로 수신: DFS {len(msg.poses)} steps, '
            f'고유 샘플링 포인트 {len(unique_poses)}개'
        )
        self.try_start_route()

    def try_start_route(self):
        if self.route_started or not self.sampling_poses:
            return
        if not self.nav_client.server_is_ready():
            self.get_logger().info(
                'Nav2 navigate_to_pose 서버 대기 중...',
                throttle_duration_sec=5.0,
            )
            return

        self.route_started = True
        self.server_timer.cancel()
        self.server_timer = None
        self.get_logger().info(
            f'샘플링 주행 시작: {len(self.sampling_poses)}개 포인트, '
            f'도착마다 {self.dwell_seconds:.1f}초 정지'
        )
        self.send_current_goal()

    def send_current_goal(self):
        if self.current_index >= len(self.sampling_poses):
            self.get_logger().info('=== 모든 샘플링 포인트 주행 완료 ===')
            return

        source_pose = self.sampling_poses[self.current_index]
        target_x = source_pose.pose.position.x
        target_y = source_pose.pose.position.y

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = source_pose.header.frame_id or 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = target_x
        goal.pose.pose.position.y = target_y

        if self.current_index + 1 < len(self.sampling_poses):
            next_pose = self.sampling_poses[self.current_index + 1]
            yaw = math.atan2(
                next_pose.pose.position.y - target_y,
                next_pose.pose.position.x - target_x,
            )
            goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            goal.pose.pose.orientation.w = 1.0

        self.goal_in_progress = True
        self.get_logger().info(
            f'이동 [{self.current_index + 1}/{len(self.sampling_poses)}] '
            f'-> ({target_x:.2f}, {target_y:.2f})'
        )
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.goal_in_progress = False
            self.handle_navigation_failure('goal rejected')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        self.goal_in_progress = False
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.handle_navigation_failure(f'status={status}')
            return

        self.retry_count = 0
        self.get_logger().info(
            f'도착 [{self.current_index + 1}/{len(self.sampling_poses)}] - '
            f'{self.dwell_seconds:.1f}초간 샘플링 정지'
        )
        self.dwell_timer = self.create_timer(
            self.dwell_seconds, self.finish_dwell
        )

    def finish_dwell(self):
        if self.dwell_timer is not None:
            self.dwell_timer.cancel()
            self.dwell_timer = None
        self.current_index += 1
        self.send_current_goal()

    def handle_navigation_failure(self, reason):
        self.retry_count += 1
        if self.retry_count <= self.max_retries:
            self.get_logger().warning(
                f'포인트 {self.current_index + 1} 이동 실패({reason}) - '
                f'재시도 {self.retry_count}/{self.max_retries}'
            )
        else:
            self.get_logger().warning(
                f'포인트 {self.current_index + 1} 이동 실패({reason}) - '
                '재시도 한계 초과, 다음 포인트로 이동'
            )
            self.retry_count = 0
            self.current_index += 1
        self.send_current_goal()


def main(args=None):
    rclpy.init(args=args)
    node = GasGridSamplerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
