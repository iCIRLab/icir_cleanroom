"""Gas-history state, calculations, and compatibility API."""

from __future__ import annotations

import math

import numpy as np


class GasHistory:
    VERSION = 3

    def __init__(self, map_msg, hazard_threshold, recent_alpha=0.5):
        self.width = int(map_msg.info.width)
        self.height = int(map_msg.info.height)
        self.resolution = float(map_msg.info.resolution)
        self.origin_x = float(map_msg.info.origin.position.x)
        self.origin_y = float(map_msg.info.origin.position.y)
        self.hazard_threshold = float(hazard_threshold)
        self.recent_alpha = float(recent_alpha)
        if (not math.isfinite(self.recent_alpha) or
                not 0.0 < self.recent_alpha <= 1.0):
            raise ValueError('recent_alpha must be in (0, 1]')
        self.free = (
            np.asarray(map_msg.data).reshape(self.height, self.width) == 0)
        self.records = {}
        self.history_mean = np.zeros((self.height, self.width), dtype=float)
        self.history_recent = np.zeros(
            (self.height, self.width), dtype=float)
        self.history_count = 0
        self.confirmed_peak_events = {}
        self.confirmed_event_sequence = 0

    @staticmethod
    def _key(row, col):
        return f'{int(row)},{int(col)}'

    def cell_center(self, row, col):
        return (
            self.origin_x + (float(col) + 0.5) * self.resolution,
            self.origin_y + (float(row) + 0.5) * self.resolution,
        )

    def _validate_cell(self, row, col, require_free=False):
        row, col = int(row), int(col)
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError('history cell is outside the map')
        if require_free and not bool(self.free[row, col]):
            raise ValueError('history cell is not free')
        return row, col

    @staticmethod
    def _finite_timestamp(timestamp, name='timestamp'):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError(f'{name} must be finite')
        return timestamp

    @staticmethod
    def _clamped_value(value, name='value'):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        return min(1.0, max(0.0, value))

    @staticmethod
    def _record_defaults(row, col):
        return {
            'row': int(row), 'col': int(col), 'historical_max': 0.0,
            'detection_count': 0, 'last_detected': None,
            'last_visited': None, 'last_phase': '',
            'visit_count': 0,
            'lrs_visit_count': 0, 'lrs_last_visited': None,
            'lrs_value_mean': 0.0, 'lrs_value_m2': 0.0,
        }

    def update(self, row, col, value, phase, timestamp):
        row, col = self._validate_cell(row, col)
        key = self._key(row, col)
        value = self._clamped_value(value)
        timestamp = self._finite_timestamp(timestamp)
        phase = str(phase)
        record = self.records.setdefault(
            key, self._record_defaults(row, col))
        record['historical_max'] = max(
            float(record['historical_max']), value)
        record['last_visited'] = timestamp
        record['last_phase'] = phase
        record['visit_count'] = int(record.get('visit_count', 0)) + 1
        if value >= self.hazard_threshold:
            record['detection_count'] = int(record['detection_count']) + 1
            record['last_detected'] = timestamp
        if phase.upper() == 'LRS':
            count = int(record.get('lrs_visit_count', 0)) + 1
            old_mean = float(record.get('lrs_value_mean', 0.0))
            delta = value - old_mean
            new_mean = old_mean + delta / float(count)
            record['lrs_value_m2'] = (
                float(record.get('lrs_value_m2', 0.0)) +
                delta * (value - new_mean))
            record['lrs_value_mean'] = new_mean
            record['lrs_visit_count'] = count
            record['lrs_last_visited'] = timestamp
        return record

    def accumulate_distribution(self, variable_values, variable_cells):
        """Add one final, linear GMRF field to the equal-weight history."""
        values = np.asarray(variable_values, dtype=float)
        cells = np.asarray(variable_cells, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != 2:
            raise ValueError('variable_cells must contain (row, col) pairs')
        if values.ndim != 1 or len(values) != len(cells):
            raise ValueError(
                'variable_values must match the number of variable cells')
        if not np.all(np.isfinite(values)):
            raise ValueError('history distribution contains non-finite values')
        if (np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= self.height) or
                np.any(cells[:, 1] < 0) or
                np.any(cells[:, 1] >= self.width)):
            raise ValueError('history distribution cell is outside the map')

        snapshot = np.zeros((self.height, self.width), dtype=float)
        snapshot[cells[:, 0], cells[:, 1]] = np.clip(values, 0.0, 1.0)
        next_count = self.history_count + 1
        self.history_mean += (
            snapshot - self.history_mean) / float(next_count)
        self.history_recent = (
            (1.0 - self.recent_alpha) * self.history_recent +
            self.recent_alpha * snapshot)
        self.history_mean[~self.free] = 0.0
        self.history_recent[~self.free] = 0.0
        self.history_count = next_count
        return self.history_count

    def record_confirmed_peak(self, event_id, row, col, value, timestamp):
        """Record exactly one confirmed peak for an independent gas event.

        Returns ``True`` for a newly inserted event and ``False`` when the
        event id was already present. Duplicate calls never increment the
        recurrence statistics.
        """
        event_id = str(event_id)
        if not event_id:
            raise ValueError('event_id must not be empty')
        if event_id in self.confirmed_peak_events:
            return False
        row, col = self._validate_cell(row, col, require_free=True)
        value = self._clamped_value(value)
        timestamp = self._finite_timestamp(timestamp)
        self.confirmed_event_sequence += 1
        self.confirmed_peak_events[event_id] = {
            'event_id': event_id,
            'sequence': self.confirmed_event_sequence,
            'row': row,
            'col': col,
            'value': value,
            'timestamp': timestamp,
        }
        return True

    def latest_confirmed_peak(self):
        """Return the newest confirmed peak, or ``None`` when none exists."""
        if not self.confirmed_peak_events:
            return None
        event = max(
            self.confirmed_peak_events.values(),
            key=lambda item: (
                int(item['sequence']), str(item['event_id'])))
        result = dict(event)
        result['x'], result['y'] = self.cell_center(
            result['row'], result['col'])
        return result

    @staticmethod
    def _validate_cells(cells):
        cells = np.asarray(cells, dtype=np.int64)
        if cells.ndim != 2 or cells.shape[1] != 2:
            raise ValueError('cells must contain (row, col) pairs')
        return cells

    def recurrence_mass(self, cells, event_half_life=10.0,
                        kernel_sigma=5.0):
        """Return decayed confirmed-event mass at the requested cells.

        Event age is measured in confirmed gas events, not wall-clock time.
        A Gaussian spatial kernel lets nearby LRS waypoints benefit from a
        confirmed peak without treating unvisited/background cells as
        negative observations.
        """
        cells = self._validate_cells(cells)
        event_half_life = float(event_half_life)
        kernel_sigma = float(kernel_sigma)
        if not math.isfinite(event_half_life) or event_half_life <= 0.0:
            raise ValueError('event_half_life must be positive')
        if not math.isfinite(kernel_sigma) or kernel_sigma <= 0.0:
            raise ValueError('kernel_sigma must be positive')
        if len(cells) == 0 or not self.confirmed_peak_events:
            return np.zeros(len(cells), dtype=float)
        if (np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= self.height) or
                np.any(cells[:, 1] < 0) or
                np.any(cells[:, 1] >= self.width)):
            raise ValueError('reward cell is outside the map')

        query_x = self.origin_x + (
            cells[:, 1].astype(float) + 0.5) * self.resolution
        query_y = self.origin_y + (
            cells[:, 0].astype(float) + 0.5) * self.resolution
        mass = np.zeros(len(cells), dtype=float)
        denominator = 2.0 * kernel_sigma * kernel_sigma
        for event in self.confirmed_peak_events.values():
            event_x, event_y = self.cell_center(
                event['row'], event['col'])
            age = max(
                0, self.confirmed_event_sequence - int(event['sequence']))
            decay = math.pow(2.0, -float(age) / event_half_life)
            squared_distance = (
                (query_x - event_x) ** 2 + (query_y - event_y) ** 2)
            mass += decay * np.exp(-squared_distance / denominator)
        return mass

    def recurrence_probability(self, cells, event_half_life=10.0,
                               kernel_sigma=5.0):
        """Convert recurrence mass into a bounded ``0..1`` score."""
        mass = self.recurrence_mass(
            cells, event_half_life, kernel_sigma)
        return np.clip(-np.expm1(-mass), 0.0, 1.0)

    def confirmed_regions(self, top_k=None, merge_radius=5.0,
                          event_half_life=10.0):
        """Group confirmed events into distinct, deterministic regions.

        The result contains a representative peak for each connected group,
        recurrence score, event count, and ``is_latest`` so the controller can
        reserve the latest confirmed peak independently from older Top-K
        regions.
        """
        merge_radius = float(merge_radius)
        event_half_life = float(event_half_life)
        if not math.isfinite(merge_radius) or merge_radius < 0.0:
            raise ValueError('merge_radius must be non-negative')
        if not math.isfinite(event_half_life) or event_half_life <= 0.0:
            raise ValueError('event_half_life must be positive')
        events = sorted(
            self.confirmed_peak_events.values(),
            key=lambda item: (int(item['sequence']), str(item['event_id'])))
        if not events:
            return []

        parent = list(range(len(events)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first, second):
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(
                    first_root, second_root)

        centers = [self.cell_center(item['row'], item['col'])
                   for item in events]
        for first in range(len(events)):
            for second in range(first + 1, len(events)):
                separation = math.hypot(
                    centers[first][0] - centers[second][0],
                    centers[first][1] - centers[second][1])
                if separation <= merge_radius:
                    union(first, second)

        groups = {}
        for index, event in enumerate(events):
            groups.setdefault(find(index), []).append(event)
        latest_sequence = self.confirmed_event_sequence
        regions = []
        for members in groups.values():
            representative = max(
                members,
                key=lambda item: (
                    int(item['sequence']), float(item['value']),
                    -int(item['row']), -int(item['col'])))
            recurrence = sum(
                math.pow(
                    2.0,
                    -float(latest_sequence - int(item['sequence'])) /
                    event_half_life)
                for item in members)
            row, col = int(representative['row']), int(representative['col'])
            x, y = self.cell_center(row, col)
            member_sequences = [int(item['sequence']) for item in members]
            regions.append({
                'row': row, 'col': col, 'x': x, 'y': y,
                'event_count': len(members),
                'recurrence_mass': float(recurrence),
                'peak_value': max(float(item['value']) for item in members),
                'latest_value': float(representative['value']),
                'last_confirmed': max(
                    float(item['timestamp']) for item in members),
                'latest_event_id': str(representative['event_id']),
                'is_latest': latest_sequence in member_sequences,
            })
        regions.sort(key=lambda item: (
            not bool(item['is_latest']),
            -float(item['recurrence_mass']),
            -float(item['peak_value']),
            -float(item['last_confirmed']),
            int(item['row']), int(item['col'])))
        if top_k is None:
            return regions
        top_k = max(0, int(top_k))
        return regions[:top_k]

    def lrs_reward_components(self, cells, now, event_half_life=10.0,
                              kernel_sigma=5.0,
                              staleness_horizon=None):
        """Return normalized LRS reward evidence for ``(row, col)`` cells.

        ``recurrence`` comes only from independently confirmed peaks;
        ``severity`` is the recent event-level GMRF EMA; ``staleness`` and
        ``uncertainty`` come only from actual LRS visits. Consequently an
        unobserved background-prior estimate is never counted as evidence that
        gas was absent.
        """
        cells = self._validate_cells(cells)
        now = self._finite_timestamp(now, 'now')
        if (np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= self.height) or
                np.any(cells[:, 1] < 0) or
                np.any(cells[:, 1] >= self.width)):
            raise ValueError('reward cell is outside the map')
        if staleness_horizon is not None:
            staleness_horizon = float(staleness_horizon)
            if (not math.isfinite(staleness_horizon) or
                    staleness_horizon <= 0.0):
                raise ValueError('staleness_horizon must be positive')

        recurrence_mass = self.recurrence_mass(
            cells, event_half_life, kernel_sigma)
        recurrence = np.clip(-np.expm1(-recurrence_mass), 0.0, 1.0)
        severity = np.clip(
            self.history_recent[cells[:, 0], cells[:, 1]], 0.0, 1.0)
        counts = np.zeros(len(cells), dtype=np.int64)
        last_visits = np.full(len(cells), np.nan, dtype=float)
        for index, (row, col) in enumerate(cells):
            record = self.records.get(self._key(row, col))
            if record is None:
                continue
            counts[index] = max(0, int(record.get('lrs_visit_count', 0)))
            last = record.get('lrs_last_visited')
            if last is not None:
                last_visits[index] = float(last)

        visited = np.isfinite(last_visits)
        elapsed = np.zeros(len(cells), dtype=float)
        elapsed[visited] = np.maximum(0.0, now - last_visits[visited])
        staleness = np.ones(len(cells), dtype=float)
        if np.any(visited):
            if staleness_horizon is not None:
                staleness[visited] = np.clip(
                    elapsed[visited] / staleness_horizon, 0.0, 1.0)
            else:
                maximum = float(np.max(elapsed[visited]))
                if maximum > 0.0:
                    staleness[visited] = np.clip(
                        elapsed[visited] / maximum, 0.0, 1.0)
                else:
                    staleness[visited] = 0.0
        uncertainty = 1.0 / np.sqrt(counts.astype(float) + 1.0)
        return {
            'recurrence_mass': recurrence_mass,
            'recurrence': recurrence,
            'severity': severity,
            'staleness': staleness,
            'uncertainty': np.clip(uncertainty, 0.0, 1.0),
            'visit_count': counts,
        }

    def lrs_staleness_horizon(self, cells, now, minimum=1.0):
        """Return a stable age scale derived only from active LRS cells."""
        cells = self._validate_cells(cells)
        now = self._finite_timestamp(now, 'now')
        minimum = float(minimum)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError('minimum staleness horizon must be positive')
        elapsed = []
        for row, col in cells:
            record = self.records.get(self._key(row, col))
            if record is None or record.get('lrs_last_visited') is None:
                continue
            last = self._finite_timestamp(
                record['lrs_last_visited'], 'lrs_last_visited')
            elapsed.append(max(0.0, now - last))
        return max([minimum] + elapsed)

    def lrs_rewards(self, cells, now, recurrence_weight=0.40,
                    severity_weight=0.20, staleness_weight=0.25,
                    uncertainty_weight=0.15, event_half_life=10.0,
                    kernel_sigma=5.0, staleness_horizon=None):
        """Return reward plus its components, all bounded to ``0..1``."""
        weights = np.asarray([
            recurrence_weight, severity_weight, staleness_weight,
            uncertainty_weight], dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError('LRS reward weights must be finite and non-negative')
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError('at least one LRS reward weight must be positive')
        weights /= total
        result = self.lrs_reward_components(
            cells, now, event_half_life, kernel_sigma, staleness_horizon)
        result['reward'] = np.clip(
            weights[0] * result['recurrence'] +
            weights[1] * result['severity'] +
            weights[2] * result['staleness'] +
            weights[3] * result['uncertainty'], 0.0, 1.0)
        return result

    def selected_regions(self, top_k, merge_radius):
        """Return GMRF-history peaks backed by hazardous measurements."""
        eligible = [
            record for record in self.records.values()
            if float(record['historical_max']) >= self.hazard_threshold
        ]
        eligible.sort(key=lambda record: (
            -float(record['historical_max']),
            -int(record['detection_count']),
            -float(record['last_detected'] or 0.0),
            int(record['row']), int(record['col'])))
        clusters = []
        assigned = set()
        for index, record in enumerate(eligible):
            if index in assigned:
                continue
            seed_x, seed_y = self.cell_center(record['row'], record['col'])
            members = []
            for other_index, other in enumerate(eligible):
                if other_index in assigned:
                    continue
                other_x, other_y = self.cell_center(
                    other['row'], other['col'])
                if (math.hypot(seed_x - other_x, seed_y - other_y) <=
                        merge_radius):
                    assigned.add(other_index)
                    members.append(other)
            clusters.append(members)

        free_rows, free_cols = np.nonzero(self.free)
        candidates = []
        used_peaks = set()
        for members in clusters:
            peak_row = int(members[0]['row'])
            peak_col = int(members[0]['col'])
            peak_value = 0.0
            if self.history_count > 0 and len(free_rows):
                verified = np.zeros(len(free_rows), dtype=bool)
                for member in members:
                    measured_x, measured_y = self.cell_center(
                        member['row'], member['col'])
                    centers_x = self.origin_x + (
                        free_cols.astype(float) + 0.5) * self.resolution
                    centers_y = self.origin_y + (
                        free_rows.astype(float) + 0.5) * self.resolution
                    verified |= (
                        (centers_x - measured_x) ** 2 +
                        (centers_y - measured_y) ** 2 <= merge_radius ** 2)
                verified_indices = np.flatnonzero(verified)
                if len(verified_indices):
                    best = min(
                        verified_indices,
                        key=lambda item: (
                            -float(self.history_mean[
                                free_rows[item], free_cols[item]]),
                            int(free_rows[item]), int(free_cols[item])))
                    best_value = float(
                        self.history_mean[free_rows[best], free_cols[best]])
                    if best_value > 0.0:
                        peak_row = int(free_rows[best])
                        peak_col = int(free_cols[best])
                        peak_value = best_value
            peak = (peak_row, peak_col)
            if peak in used_peaks:
                continue
            used_peaks.add(peak)
            x, y = self.cell_center(peak_row, peak_col)
            last_detected = max(
                float(item['last_detected'] or 0.0) for item in members)
            last_visited = max(
                float(item['last_visited'] or 0.0) for item in members)
            candidates.append({
                'row': peak_row,
                'col': peak_col,
                'x': x,
                'y': y,
                'historical_mean': peak_value,
                'historical_max': max(
                    float(item['historical_max']) for item in members),
                'detection_count': sum(
                    int(item['detection_count']) for item in members),
                'last_detected': last_detected,
                'last_visited': last_visited,
                'last_phase': str(members[0].get('last_phase', '')),
            })
        candidates.sort(key=lambda record: (
            -float(record['historical_mean']),
            -int(record['detection_count']),
            -float(record['last_detected'] or 0.0),
            int(record['row']), int(record['col'])))
        return candidates[:int(top_k)]




    def reset(self):
        self.records.clear()
        self.history_mean.fill(0.0)
        self.history_recent.fill(0.0)
        self.history_count = 0
        self.confirmed_peak_events.clear()
        self.confirmed_event_sequence = 0

