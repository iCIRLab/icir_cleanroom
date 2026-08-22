"""Navigation progress and asynchronous goal identity management."""

from ..models import NavigationState


class NavigationManager:
    def __init__(self, state=None):
        self.state = state or NavigationState()
        self._goal_generation = 0
        self.active_goal_handle = None

    def reset_progress(self):
        self.invalidate_goal()
        self.state.reset_progress()

    def issue_goal(self):
        self._goal_generation += 1
        self.state.active_goal_id = self._goal_generation
        return self._goal_generation

    def accepts(self, generation):
        return self.state.active_goal_id == int(generation)

    def set_goal_handle(self, generation, handle):
        if not self.accepts(generation):
            return False
        self.active_goal_handle = handle
        return True

    def invalidate_goal(self):
        self._goal_generation += 1
        self.state.active_goal_id = None
        self.active_goal_handle = None

    def cancel_active_goal(self):
        handle = self.active_goal_handle
        self.invalidate_goal()
        if handle is not None:
            return handle.cancel_goal_async()
        return None


__all__ = ['NavigationManager']
