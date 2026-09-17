"""Single-cell, repeatedly replanned HRS DD-UCB workflow."""

from .hrs_run_logging import record_hrs
from ..application.hrs_methods import HrsSession
from ..application.hrs_termination import attempts_exhausted


class HrsWorkflow:
    METHODS = [
        'available_variables', 'build_candidates', 'start_hrs_planning',
        'finish_hrs_cycle', 'finish_hrs_search']

    def __init__(self, controller):
        self.controller = controller

    def available_variables(self):
        return self.controller.hrs_manager.available_variables(
            len(self.controller.gmrf.var_cells),
            self.controller.sampled_variables,
            self.controller.navigation_goal_variables)

    def build_candidates(self):
        session = getattr(self.controller, 'hrs_session', None)
        sampled = self.controller.sampled_variables
        if session is not None and session.stage == 'INITIAL' and session.method.initial in ('lrs_max', 'lrs_centroid'):
            sampled = set()
        return self.controller.hrs_manager.build_candidates(
            self.controller.gmrf,
            sampled,
            self.controller.hrs_ucb_k,
            self.controller.navigation_goal_variables)

    def start_hrs_planning(self):
        if self.controller.planning_executor.active_task is not None:
            raise RuntimeError('another route-planning job is already active')
        self.controller.publish_phase('HRS_PLANNING')
        current_xy = (
            self.controller.latest_pose.pose.position.x,
            self.controller.latest_pose.pose.position.y)
        try:
            candidates = self.controller.build_candidates()
            session = getattr(self.controller, 'hrs_session', None)
            if session is None:
                session = self.controller.hrs_session = HrsSession(
                    getattr(self.controller, 'method', 'M3'))
            scored_candidates, selected = session.select(
                candidates, gmrf=getattr(self.controller, 'gmrf', None), current_xy=current_xy,
                oracle=self.controller.sampling_distance,
                distance_weight=self.controller.hrs_distance_weight,
                position_valid=getattr(self.controller, 'is_navigation_goal_position', None))
        except (TypeError, ValueError) as error:
            self.controller.get_logger().error(
                f'HRS candidate selection failed: {error}')
            self.finish_hrs_search(f'candidate selection failed: {error}')
            return

        previous = set(self.controller.hrs_candidate_variables)
        current = {
            candidate.variable for candidate in scored_candidates}
        self.controller.hrs_candidate_variables = current
        added = current - previous
        removed = previous - current
        self.controller.publish_candidates(scored_candidates, (), selected)
        self.controller.get_logger().info(
            f'HRS iteration {self.controller.hrs_cycles + 1}: '
            f'robot_pose=({current_xy[0]:.3f},{current_xy[1]:.3f}), '
            f'ucb_k={float(self.controller.hrs_ucb_k):.3f}, '
            f'distance_weight='
            f'{float(self.controller.hrs_distance_weight):.4f}, '
            f'candidates={len(scored_candidates)}, '
            f'candidate_delta=+{len(added)}/-{len(removed)}')

        if selected is None:
            self.controller.active_hrs_target = None
            self.controller.hrs_route_targets = []
            self.controller.publish_empty_hrs_route()
            reason = ('spiral exhausted' if session.stage == 'SEARCH' and
                      session.method.search == 'spiral' else
                      'no eligible unmeasured HRS cells remain')
            self.finish_hrs_search(reason)
            return

        self.controller.hrs_route_targets = [selected]
        record_hrs(self.controller, 'target_selected', variable=selected.variable,
                   x=selected.x, y=selected.y, score=selected.score,
                   requested_xy=session.requested_xy, attempt_count=session.attempt_count,
                   search_iterations=session.search_iterations)
        self.controller.active_hrs_target = selected
        self.controller.current_index = 0
        self.controller.retry = 0
        self.controller.hrs_cycle_started_ns = (
            self.controller.get_clock().now().nanoseconds)
        self.controller.publish_hrs_route((selected,))
        self.controller.publish_phase('HRS_NAVIGATION')
        self.controller.get_logger().info(
            f'HRS {session.method_id}/{session.stage} target selected: variable={selected.variable}, '
            f'cell=({selected.row},{selected.col}), '
            f'position=({selected.x:.3f},{selected.y:.3f}), '
            f'mu={selected.mean:.6f}, '
            f'sigma={max(selected.variance, 0.0) ** 0.5:.6f}, '
            f'ucb={selected.ucb:.6f}, '
            f'normalized_ucb={selected.normalized_ucb:.6f}, '
            f'distance={selected.distance:.3f}, '
            f'normalized_distance={selected.normalized_distance:.6f}, '
            f'strategy_score={selected.score:.6f}')
        self.controller.send_current_goal()

    def finish_hrs_search(self, reason):
        record_hrs(self.controller, 'finish', outcome='failed', reason=reason)
        self.controller.get_logger().warning(f'HRS terminated: {reason}')
        if self.controller.repeat_after_hrs:
            self.controller.start_source_transition(f'HRS terminated: {reason}')
        else:
            self.controller.complete_mapping(f'HRS terminated: {reason}')

    def finish_hrs_cycle(self):
        actual_seconds = 0.0
        if self.controller.hrs_cycle_started_ns is not None:
            actual_seconds = (
                (self.controller.get_clock().now().nanoseconds -
                 self.controller.hrs_cycle_started_ns) * 1.0e-9)
        target = self.controller.active_hrs_target
        self.controller.hrs_cycles += 1
        self.controller.hrs_cycles_in_alert += 1
        self.controller.get_logger().info(
            f'=== HRS iteration {self.controller.hrs_cycles} complete: '
            f'target={None if target is None else target.variable}, '
            f'actual={actual_seconds:.3f}s, action=recompute_candidates ===')
        map_updated = self.controller.finalize_hrs_gmrf_batch(
            f'HRS iteration {self.controller.hrs_cycles}')
        self.controller.publish_hrs_status()
        self.controller.persist_history(
            f'HRS iteration {self.controller.hrs_cycles}')
        self.controller.active_hrs_target = None
        self.controller.hrs_route_targets = []
        self.controller.hrs_cycle_started_ns = None
        if not map_updated:
            self.finish_hrs_search('GMRF update failed')
            return
        if attempts_exhausted(self.controller.hrs_cycles_in_alert,
                              int(self.controller.hrs_max_cycles_per_alert)):
            self.finish_hrs_search(
                f'maximum {self.controller.hrs_cycles_in_alert} '
                'HRS target attempts reached')
            return
        session = getattr(self.controller, 'hrs_session', None)
        if session is not None and session.stage == 'SEARCH' and session.stalled():
            self.finish_hrs_search(
                f'no relative improvement in {session.no_improvement_streak} '
                'consecutive HRS attempts')
            return
        self.controller.start_hrs_planning()


__all__ = ['HrsWorkflow']
