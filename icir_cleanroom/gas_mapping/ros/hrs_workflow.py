"""Single-cell, repeatedly replanned GMRF centroid HRS workflow."""


class HrsWorkflow:
    METHODS = [
        'available_variables', 'begin_hrs_measurement_timing',
        'finish_hrs_measurement_timing', 'start_hrs_planning',
        'confirm_hrs_response', 'finish_hrs_cycle', 'finish_hrs_search']

    def __init__(self, controller):
        self.controller = controller

    def available_variables(self):
        return self.controller.hrs_manager.available_variables(
            len(self.controller.gmrf.var_cells),
            self.controller.sampled_variables,
            self.controller.navigation_goal_variables)

    def begin_hrs_measurement_timing(self):
        """Start one timer spanning the complete HRS measurement session."""
        if self.controller.hrs_search_started_ns is not None:
            return
        self.controller.hrs_search_started_ns = (
            self.controller.get_clock().now().nanoseconds)
        self.controller.get_logger().info(
            f'=== HRS 전체 측정 시작: LRS lap={self.controller.lrs_lap} ===')

    def finish_hrs_measurement_timing(self, outcome):
        """Log and clear the complete HRS measurement-session timer."""
        started_ns = getattr(self.controller, 'hrs_search_started_ns', None)
        if started_ns is None:
            return None
        finished_ns = self.controller.get_clock().now().nanoseconds
        elapsed_seconds = max(0.0, (finished_ns - started_ns) * 1.0e-9)
        self.controller.hrs_search_started_ns = None
        self.controller.get_logger().info(
            f'=== HRS 전체 측정 종료: outcome={outcome}, '
            f'elapsed={elapsed_seconds:.3f}s ===')
        return elapsed_seconds

    def start_hrs_planning(self):
        if self.controller.planning_executor.active_task is not None:
            raise RuntimeError('another route-planning job is already active')
        self.controller.publish_phase('HRS_PLANNING')
        self.begin_hrs_measurement_timing()
        try:
            selection = self.controller.hrs_manager.select_target(
                self.controller.gmrf,
                self.controller.sampled_variables,
                self.controller.hrs_centroid_threshold,
                self.controller.navigation_goal_variables)
        except (TypeError, ValueError) as error:
            self.controller.get_logger().error(
                f'HRS weighted-centroid selection failed: {error}')
            self.finish_hrs_search('weighted-centroid selection failed')
            return

        self.controller.hrs_centroid = selection.centroid
        self.controller.publish_centroid(
            selection.centroid, selection.target)
        target = selection.target
        if selection.centroid is None:
            self.controller.active_hrs_target = None
            self.controller.publish_empty_hrs_route()
            self.finish_hrs_search(
                'no GMRF concentration above centroid threshold')
            return
        if target is None:
            self.controller.active_hrs_target = None
            self.controller.publish_empty_hrs_route()
            self.finish_hrs_search(
                'no unmeasured reachable navigation cell remains')
            return

        self.controller.active_hrs_target = target
        self.controller.current_index = 0
        self.controller.retry = 0
        self.controller.hrs_cycle_started_ns = (
            self.controller.get_clock().now().nanoseconds)
        self.controller.publish_hrs_route(target)
        self.controller.publish_phase('HRS_NAVIGATION')
        self.controller.get_logger().info(
            f'HRS weighted-centroid target: '
            f'centroid=({selection.centroid[0]:.3f},'
            f'{selection.centroid[1]:.3f}), '
            f'variable={target.variable}, cell=({target.row},{target.col}), '
            f'position=({target.x:.3f},{target.y:.3f}), '
            f'mu={target.mean:.6f}, weight={target.weight:.6f}, '
            f'centroid_distance={target.centroid_distance:.3f}')
        self.controller.send_current_goal()

    def confirm_hrs_response(self, variable, value, timestamp):
        threshold = float(self.controller.hrs_response_threshold)
        if not self.controller.hrs_manager.reached_response_threshold(
                value, threshold):
            self.controller.get_logger().info(
                f'HRS continue: measured={float(value):.6f} < '
                f'response_threshold={threshold:.6f}; '
                'GMRF and weighted centroid will be recomputed')
            return False

        row, col = self.controller.gmrf.var_cells[int(variable)]
        try:
            inserted = self.controller.history.record_confirmed_event(
                self.controller.current_event_id, int(row), int(col),
                float(value), threshold, float(timestamp),
                method='threshold_crossing')
        except (TypeError, ValueError) as error:
            self.controller.get_logger().error(
                f'HRS 확정 검출 이벤트 기록 실패: {error}')
            inserted = False
        if not inserted:
            self.controller.get_logger().warning(
                'HRS 검출 이벤트가 이미 기록되었거나 저장할 수 없습니다: '
                f'event={self.controller.current_event_id}')
        self.controller.publish_history()
        self.controller.persist_history('successful HRS detection')
        if self.controller.hrs_gmrf_dirty:
            if not self.controller.finalize_hrs_gmrf_batch(
                    'successful HRS detection'):
                self.controller.get_logger().warning(
                    '확정 검출은 유지하지만 최종 GMRF 복구에는 실패했습니다')
        self.controller.get_logger().warning(
            f'HRS successful detection: cell=({int(row)},{int(col)}), '
            f'measured={float(value):.6f} >= threshold={threshold:.6f}')
        self.finish_hrs_measurement_timing('response threshold reached')
        self.controller.start_source_transition(
            f'successful HRS detection at ({int(row)},{int(col)}), '
            f'value={float(value):.4f}, threshold={threshold:.4f}')
        return True

    def finish_hrs_search(self, reason):
        self.controller.get_logger().warning(f'HRS terminated: {reason}')
        self.finish_hrs_measurement_timing(reason)
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
            f'=== HRS centroid iteration {self.controller.hrs_cycles} '
            f'complete: target='
            f'{None if target is None else target.variable}, '
            f'actual={actual_seconds:.3f}s, action=recompute_centroid ===')
        map_updated = self.controller.finalize_hrs_gmrf_batch(
            f'HRS centroid iteration {self.controller.hrs_cycles}')
        self.controller.publish_hrs_status()
        self.controller.persist_history(
            f'HRS centroid iteration {self.controller.hrs_cycles}')
        self.controller.active_hrs_target = None
        self.controller.hrs_cycle_started_ns = None
        if not map_updated:
            self.finish_hrs_search('GMRF update failed')
            return
        if (self.controller.hrs_cycles_in_alert >=
                int(self.controller.hrs_max_cycles_per_alert)):
            self.finish_hrs_search(
                f'maximum {self.controller.hrs_cycles_in_alert} '
                'HRS iterations reached')
            return
        self.start_hrs_planning()


__all__ = ['HrsWorkflow']
