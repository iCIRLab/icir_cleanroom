"""Nav2 action transport with token-aware callbacks."""

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


class Nav2Client:
    def __init__(self, node, navigation_manager):
        self._manager = navigation_manager
        self._client = ActionClient(node, NavigateToPose, 'navigate_to_pose')

    def server_is_ready(self):
        return self._client.server_is_ready()

    def send(self, pose, generation, on_success, on_failure):
        goal = NavigateToPose.Goal()
        goal.pose = pose
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
        on_success()


__all__ = ['Nav2Client']
