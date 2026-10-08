"""Nav2 base-position goals and map-frame robot poses for gas observations."""

import copy
import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class Nav2Client:
    # Use the same body frame as Nav2, without any sensor-mount offset.
    BASE_FRAME = 'base_link'
    MAP_FRAME = 'map'
    MAX_POSE_AGE_SECONDS = 0.5
    # Gazebo's TF stream can lead the last received /clock tick (~0.1 s).
    # Allow bounded delivery skew, while rejecting genuinely future poses.
    MAX_POSE_LEAD_SECONDS = 0.2

    def __init__(self, node, navigation_manager):
        self._node = node
        self._manager = navigation_manager
        self._client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
        self._tf = Buffer(node=node)
        self._tf_listener = TransformListener(self._tf, node)

    def server_is_ready(self):
        return self._client.server_is_ready() and self.robot_pose() is not None

    def robot_pose(self):
        """Return a fresh map -> base_link pose, never a sensor pose fallback."""
        try:
            transform = self._tf.lookup_transform(
                self.MAP_FRAME, self.BASE_FRAME, Time())
        except TransformException:
            return None
        stamp = transform.header.stamp.sec * 10**9 + transform.header.stamp.nanosec
        age = (self._node.get_clock().now().nanoseconds - stamp) * 1.e-9
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        if (not -self.MAX_POSE_LEAD_SECONDS <= age <= self.MAX_POSE_AGE_SECONDS or
                not all(math.isfinite(v) for v in (
                    translation.x, translation.y, translation.z,
                    rotation.x, rotation.y, rotation.z, rotation.w))):
            return None
        pose = PoseStamped()
        pose.header = copy.deepcopy(transform.header)
        pose.pose.position.x = translation.x
        pose.pose.position.y = translation.y
        pose.pose.position.z = translation.z
        pose.pose.orientation = copy.deepcopy(rotation)
        return pose

    def send(self, pose, generation, on_success, on_failure):
        """Send the requested robot-centre target without offset compensation."""
        if not self._manager.accepts(generation):
            return
        goal = NavigateToPose.Goal()
        goal.pose = copy.deepcopy(pose)
        future = self._client.send_goal_async(goal)
        future.add_done_callback(
            lambda completed: self._goal_response(
                completed, generation, on_success, on_failure))

    def _goal_response(
            self, future, generation, on_success, on_failure):
        if not self._manager.accepts(generation):
            return
        try:
            handle = future.result()
        except Exception as error:
            on_failure(f'goal request exception: {error}')
            return
        if not handle.accepted:
            on_failure('goal rejected')
            return
        if not self._manager.set_goal_handle(generation, handle):
            handle.cancel_goal_async()
            return
        result = handle.get_result_async()
        result.add_done_callback(
            lambda completed: self._goal_result(
                completed, generation, on_success, on_failure))

    def _goal_result(self, future, generation, on_success, on_failure):
        if not self._manager.accepts(generation):
            return
        try:
            status = future.result().status
        except Exception as error:
            on_failure(f'goal result exception: {error}')
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            on_failure(f'status={status}')
            return
        pose = self.robot_pose()
        if pose is None:
            on_failure('robot pose unavailable after Nav2 arrival')
            return
        # The dwell session and final result must capture the body position
        # at arrival, even if the periodic pose update has not run yet.
        self._node.pose_callback(pose)
        on_success()


__all__ = ['Nav2Client']
