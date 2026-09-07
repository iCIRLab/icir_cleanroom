"""Single-cell, repeatedly replanned HRS DD-UCB workflow."""


class HrsWorkflow:
    METHODS = [
        'available_variables', 'build_candidates', 'start_hrs_planning',
        'confirm_hrs_response', 'finish_hrs_cycle', 'finish_hrs_search']

    def __init__(self, controller):
        self.controller = controller

    def available_variables(self):
        return self.controller.hrs_manager.available_variables(
            len(self.controller.gmrf.var_cells),
            self.controller.sampled_variables,
            self.controller.navigation_goal_variables)

    def build_candidates(self):
        return self.controller.hrs_manager.build_candidates(
            self.controller.gmrf,
            self.controller.sampled_variables,
            self.controller.hrs_ucb_k,
            self.controller.hrs_candidate_threshold,
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
            scored_candidates, selected = (
                self.controller.hrs_manager.select_candidate(
                    candidates, current_xy,
                    self.controller.sampling_distance.distance,
                    self.controller.hrs_distance_weight))
        except (TypeError, ValueError) as error:
            self.controller.get_logger().error(
                f'HRS candidate selection failed: {error}')
            self.finish_hrs_search('candidate selection failed')
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
            f'T_candidate_ucb='
            f'{float(self.controller.hrs_candidate_threshold):.4f}, '
            f'distance_weight='
            f'{float(self.controller.hrs_distance_weight):.4f}, '
            f'candidates={len(scored_candidates)}, '
            f'candidate_delta=+{len(added)}/-{len(removed)}')

        if selected is None:
            self.controller.active_hrs_target = None
            self.controller.hrs_route_targets = []
            self.controller.publish_empty_hrs_route()
            self.finish_hrs_search(
                'no UCB candidate above candidate threshold')
            return

        self.controller.hrs_route_targets = [selected]
        self.controller.active_hrs_target = selected
        self.controller.current_index = 0
        self.controller.retry = 0
        self.controller.hrs_cycle_started_ns = (
            self.controller.get_clock().now().nanoseconds)
        self.controller.publish_hrs_route((selected,))
        self.controller.publish_phase('HRS_NAVIGATION')
        self.controller.get_logger().info(
            f'HRS target selected: variable={selected.variable}, '
            f'cell=({selected.row},{selected.col}), '
            f'position=({selected.x:.3f},{selected.y:.3f}), '
            f'mu={selected.mean:.6f}, '
            f'sigma={max(selected.variance, 0.0) ** 0.5:.6f}, '
            f'ucb={selected.ucb:.6f}, '
            f'normalized_ucb={selected.normalized_ucb:.6f}, '
            f'distance={selected.distance:.3f}, '
            f'normalized_distance={selected.normalized_distance:.6f}, '
            f'dd_ucb_score={selected.score:.6f}')
        self.controller.send_current_goal()

    def confirm_hrs_response(self, variable, value, timestamp):
        goal_threshold = float(self.controller.hrs_response_threshold)
        if not self.controller.hrs_manager.reached_response_threshold(
                value, goal_threshold):
            self.controller.get_logger().info(
                f'HRS continue: measured={float(value):.6f} < '
                f'T_goal_concentration={goal_threshold:.6f}; '
                'the observation was added and candidates will be recomputed')
            return False

        self.controller.hrs_confirmed_variable = int(variable)
        self.controller.hrs_confirmed_value = float(value)
        self.controller.hrs_confirmed_timestamp = float(timestamp)
        row, col = self.controller.gmrf.var_cells[int(variable)]
        try:
            inserted = self.controller.history.record_confirmed_event(
                self.controller.current_event_id, int(row), int(col),
                float(value), goal_threshold, float(timestamp),
                method='threshold_crossing')
        except (TypeError, ValueError) as error:
            self.controller.get_logger().error(
                f'HRS successful detection event record failed: {error}')
            inserted = False
        if not inserted:
            self.controller.get_logger().warning(
                'HRS successful detection event was already recorded or '
                f'could not be saved: event={self.controller.current_event_id}')
        self.controller.publish_history()
        self.controller.persist_history('successful HRS detection')
        self.controller.get_logger().warning(
            f'HRS successful detection: variable={int(variable)}, '
            f'cell=({int(row)},{int(col)}), measured={float(value):.6f} >= '
            f'T_goal_concentration={goal_threshold:.6f}; terminating HRS')
        self.controller.start_source_transition(
            f'successful HRS detection at ({int(row)},{int(col)}), '
            f'value={float(value):.4f}, threshold={goal_threshold:.4f}')
        return True

    def finish_hrs_search(self, reason):
        self.controller.get_logger().warning(f'HRS terminated: {reason}')
        self.controller.return_to_lrs(f'HRS terminated: {reason}')

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
        if (self.controller.hrs_cycles_in_alert >=
                int(self.controller.hrs_max_cycles_per_alert)):
            self.finish_hrs_search(
                f'maximum {self.controller.hrs_cycles_in_alert} '
                'HRS measurements reached')
            return
        self.controller.start_hrs_planning()


__all__ = ['HrsWorkflow']
