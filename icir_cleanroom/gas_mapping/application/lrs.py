"""LRS runtime-state owner, reward preparation, and lap operations."""

import math
import time

import numpy as np

from ..models import LrsRuntimeState
from ..planning.lrs_priority import LrsRewardPoint


class LrsManager:
    def __init__(self, state=None):
        self.state = state or LrsRuntimeState()

    def begin_lap(self):
        self.state.lap += 1
        self.state.lap_max_concentration = 0.0
        self.state.hazard_detected = False
        self.state.goals.clear()
        self.state.status_values.clear()
        self.state.priority_points.clear()
        return self.state.lap

    def record_measurement(self, value, hazard_threshold):
        value = float(value)
        first_hazard = not self.state.hazard_detected and (
            value >= float(hazard_threshold))
        self.state.lap_max_concentration = max(
            self.state.lap_max_concentration, value)
        self.state.hazard_detected = (
            self.state.hazard_detected or value >= float(hazard_threshold))
        return first_hazard

    @staticmethod
    def closed_cycle_reference_length(
            points, start_xy, distance_fn=None):
        if not points:
            return 0.0
        distance_fn = distance_fn or (
            lambda first, second: math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1])))
        cycle = 0.0
        for index, first in enumerate(points):
            second = points[(index + 1) % len(points)]
            cycle += distance_fn(first, second)
        entry = min(
            distance_fn(tuple(start_xy), point) for point in points)
        return cycle + entry

    def prepare_reward_points(
            self, adjusted_points, gmrf, history, lrs_config,
            history_config, now=None):
        """Build the unique active set and all history-aware rewards."""
        latest = history.latest_confirmed_peak()
        mandatory_variable = None
        if latest is not None:
            mandatory_variable = gmrf.variable_at(
                int(latest['row']), int(latest['col']))

        active = []
        active_index_by_variable = {}
        duplicates = []
        for x, y in adjusted_points:
            variable = int(gmrf.nearest_variable(float(x), float(y)))
            if variable in active_index_by_variable:
                duplicates.append(variable)
                if variable == mandatory_variable:
                    active[active_index_by_variable[variable]] = (
                        variable, float(x), float(y))
                continue
            active_index_by_variable[variable] = len(active)
            active.append((variable, float(x), float(y)))
        if len(active) < 2:
            raise ValueError('adaptive LRS requires at least two unique cells')

        active_cells = np.asarray([
            gmrf.var_cells[variable] for variable, _, _ in active],
            dtype=np.int64)
        now = time.time() if now is None else float(now)
        horizon = history.lrs_staleness_horizon(active_cells, now)
        reward_arguments = dict(
            recurrence_weight=float(
                lrs_config.lrs_reward_recurrence_weight),
            severity_weight=float(lrs_config.lrs_reward_severity_weight),
            staleness_weight=float(lrs_config.lrs_reward_staleness_weight),
            uncertainty_weight=float(
                lrs_config.lrs_reward_uncertainty_weight),
            event_half_life=float(
                history_config.history_event_half_life),
            kernel_sigma=float(history_config.history_event_kernel_sigma),
            staleness_horizon=horizon)
        reward_data = history.lrs_rewards(
            active_cells, now, **reward_arguments)
        display_data = history.lrs_rewards(
            gmrf.var_cells, now, **reward_arguments)
        self.state.reward_values = np.asarray(
            display_data['reward'], dtype=float)

        points = []
        for index, (variable, x, y) in enumerate(active):
            row, col = gmrf.var_cells[variable]
            points.append(LrsRewardPoint(
                variable=variable, row=int(row), col=int(col),
                x=x, y=y, reward=float(reward_data['reward'][index]),
                recurrence=float(reward_data['recurrence'][index]),
                severity=float(reward_data['severity'][index]),
                staleness=float(reward_data['staleness'][index]),
                uncertainty=float(reward_data['uncertainty'][index]),
                mandatory=(variable == mandatory_variable)))
        ranked = sorted(points, key=lambda point: (
            not point.mandatory, -point.reward, point.row, point.col,
            point.variable))
        self.state.priority_candidates = ranked[
            :int(lrs_config.lrs_priority_candidate_count)]
        return points, duplicates


__all__ = ['LrsManager']
