"""Atomic JSON persistence for :class:`GasHistory`."""

import json
import math
import os
import tempfile

import numpy as np

from .model import GasHistory


class HistoryRepository:
    """Load and save history payloads without owning domain state."""

    def __init__(self, path):
        self.path = os.path.expanduser(str(path))

    @staticmethod
    def _metadata(history):
        return {
            'width': history.width, 'height': history.height,
            'resolution': history.resolution,
            'origin_x': history.origin_x, 'origin_y': history.origin_y,
        }

    @staticmethod
    def _to_payload(history):
        """Return the versioned JSON-compatible representation."""
        return {
            'version': history.VERSION,
            'map': HistoryRepository._metadata(history),
            'records': [
                history.records[key] for key in sorted(history.records)],
            'history_count': history.history_count,
            'history_mean': history.history_mean.ravel().tolist(),
            'history_recent': history.history_recent.ravel().tolist(),
            'recent_alpha': history.recent_alpha,
            'confirmed_event_sequence': history.confirmed_event_sequence,
            'confirmed_peak_events': sorted(
                history.confirmed_peak_events.values(),
                key=lambda item: (
                    int(item['sequence']), str(item['event_id']))),
        }

    @staticmethod
    def _load_payload(history, payload):
        """Validate and atomically apply a version 1, 2, or 3 payload."""
        version = int(payload.get('version', -1))
        if version not in (1, 2, history.VERSION):
            raise ValueError('unsupported gas history version')
        metadata = payload.get('map', {})
        expected = HistoryRepository._metadata(history)
        for name in ('width', 'height'):
            if int(metadata.get(name, -1)) != int(expected[name]):
                raise ValueError(f'incompatible history map {name}')
        for name in ('resolution', 'origin_x', 'origin_y'):
            if not math.isclose(
                    float(metadata.get(name, math.nan)), expected[name],
                    rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f'incompatible history map {name}')
        records = {}
        for raw in payload.get('records', []):
            row, col = history._validate_cell(raw['row'], raw['col'])
            record = history._record_defaults(row, col)
            record.update({
                'historical_max': history._clamped_value(
                    raw['historical_max'], 'historical_max'),
                'detection_count': max(
                    0, int(raw.get('detection_count', 0))),
                'last_detected': raw.get('last_detected'),
                'last_visited': raw.get('last_visited'),
                'last_phase': str(raw.get('last_phase', '')),
                'visit_count': max(0, int(raw.get('visit_count', 0))),
                'lrs_visit_count': max(
                    0, int(raw.get('lrs_visit_count', 0))),
                'lrs_last_visited': raw.get('lrs_last_visited'),
                'lrs_value_mean': history._clamped_value(
                    raw.get('lrs_value_mean', 0.0), 'lrs_value_mean'),
                'lrs_value_m2': max(
                    0.0, float(raw.get('lrs_value_m2', 0.0))),
            })
            for name in ('last_detected', 'last_visited',
                         'lrs_last_visited'):
                if record[name] is not None:
                    record[name] = history._finite_timestamp(
                        record[name], name)
            if not math.isfinite(record['lrs_value_m2']):
                raise ValueError('lrs_value_m2 must be finite')
            records[history._key(row, col)] = record

        history_mean = np.zeros((history.height, history.width), dtype=float)
        history_recent = np.zeros((history.height, history.width), dtype=float)
        history_count = 0
        if version >= 2:
            history_count = int(payload.get('history_count', 0))
            if history_count < 0:
                raise ValueError('history_count must be non-negative')
            raw_mean = np.asarray(
                payload.get('history_mean', []), dtype=float)
            if raw_mean.size != history.width * history.height:
                raise ValueError('history_mean has an incompatible size')
            if not np.all(np.isfinite(raw_mean)):
                raise ValueError('history_mean contains non-finite values')
            history_mean = np.clip(
                raw_mean.reshape(history.height, history.width), 0.0, 1.0)
            history_mean[~history.free] = 0.0
            if version == 2:
                # Version 2 has no recent EMA. Seeding it from the preserved
                # long-term mean gives migration users useful severity while
                # keeping the new confirmed-event recurrence state empty.
                history_recent = history_mean.copy()
            else:
                raw_recent = np.asarray(
                    payload.get('history_recent', []), dtype=float)
                if raw_recent.size != history.width * history.height:
                    raise ValueError(
                        'history_recent has an incompatible size')
                if not np.all(np.isfinite(raw_recent)):
                    raise ValueError(
                        'history_recent contains non-finite values')
                history_recent = np.clip(
                    raw_recent.reshape(history.height, history.width), 0.0, 1.0)
                history_recent[~history.free] = 0.0

        confirmed_peak_events = {}
        confirmed_event_sequence = 0
        if version >= 3:
            stored_sequence = int(
                payload.get('confirmed_event_sequence', 0))
            if stored_sequence < 0:
                raise ValueError(
                    'confirmed_event_sequence must be non-negative')
            sequences = set()
            for raw in payload.get('confirmed_peak_events', []):
                event_id = str(raw['event_id'])
                if not event_id:
                    raise ValueError('confirmed event id must not be empty')
                if event_id in confirmed_peak_events:
                    raise ValueError('duplicate confirmed event id')
                sequence = int(raw['sequence'])
                if sequence <= 0 or sequence in sequences:
                    raise ValueError('invalid confirmed event sequence')
                row, col = history._validate_cell(
                    raw['row'], raw['col'], require_free=True)
                confirmed_peak_events[event_id] = {
                    'event_id': event_id,
                    'sequence': sequence,
                    'row': row,
                    'col': col,
                    'value': history._clamped_value(
                        raw['value'], 'confirmed peak value'),
                    'timestamp': history._finite_timestamp(
                        raw['timestamp'], 'confirmed peak timestamp'),
                }
                sequences.add(sequence)
            maximum_sequence = max(sequences, default=0)
            if stored_sequence < maximum_sequence:
                raise ValueError(
                    'confirmed_event_sequence precedes stored events')
            confirmed_event_sequence = stored_sequence

        # Commit only after the complete payload has been validated.
        history.records = records
        history.history_mean = history_mean
        history.history_recent = history_recent
        history.history_count = history_count
        history.confirmed_peak_events = confirmed_peak_events
        history.confirmed_event_sequence = confirmed_event_sequence
        return True


    def save(self, history):
        directory = os.path.dirname(self.path) or '.'
        os.makedirs(directory, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', dir=directory, prefix='.gas_history_',
                    suffix='.json', delete=False, encoding='utf-8') as stream:
                json.dump(
                    self._to_payload(history), stream,
                    ensure_ascii=False, indent=2)
                temporary_path = stream.name
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load(self, history):
        if not os.path.exists(self.path):
            return False
        with open(self.path, 'r', encoding='utf-8') as stream:
            payload = json.load(stream)
        return self._load_payload(history, payload)

    def clear(self, history):
        history.reset()
        self.save(history)


class GasHistoryStore(GasHistory):
    """Backward-compatible facade used by the existing controller."""

    def __init__(self, path, map_msg, hazard_threshold, recent_alpha=0.5):
        super().__init__(map_msg, hazard_threshold, recent_alpha)
        self.repository = HistoryRepository(path)
        self.path = self.repository.path

    def save(self):
        self.repository.save(self)

    def load(self):
        return self.repository.load(self)

    def clear(self):
        self.repository.clear(self)


__all__ = ['GasHistoryStore', 'HistoryRepository']
