"""Per-HRS timing and outcome records, independent of selection and truth input.

Truth is used only to write post-hoc error; this class never returns a target or
changes a termination decision. Wall duration uses a monotonic clock supplied
by the caller. Missing measurements/source positions stay empty in CSV.
"""
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid


class HrsRunLog:
    def __init__(self, directory, time_basis):
        self.directory = Path(directory).expanduser() / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8])
        self.time_basis = time_basis
        self.active = None
        self.run_count = 0

    def _append(self, name, row):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory/name
        exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _clock(self, ros, wall):
        run = self.active
        if ros < run['last_ros'] or wall < run['last_wall']:
            run['clock_valid'] = False
        run['last_ros'], run['last_wall'] = ros, wall

    def event(self, kind, ros, wall, **details):
        if self.active is None:
            return
        self._clock(ros, wall)
        run = self.active
        self._append('events.csv', dict(
            run_id=run['run_id'], event=kind, ros_seconds=ros,
            wall_elapsed_seconds=wall-run['start_wall'],
            utc=datetime.now(timezone.utc).isoformat(),
            details=json.dumps(details, ensure_ascii=False, allow_nan=False)))

    def start(self, ros, wall, *, event_id, lrs_lap, source, parameters):
        if self.active is not None:
            raise RuntimeError('previous HRS record is still active')
        self.run_count += 1
        self.active = dict(
            run_id=f'{self.directory.name}_{self.run_count:04d}',
            event_id=event_id, lrs_lap=lrs_lap, source_start=source,
            parameters=parameters, start_ros=ros, start_wall=wall,
            last_ros=ros, last_wall=wall, first_ros=None, first_wall=None,
            last_measurement=None, measurement_count=0, clock_valid=True,
            start_utc=datetime.now(timezone.utc).isoformat())
        self.event('lrs_complete_hrs_start', ros, wall,
                   source=source, parameters=parameters)

    def measurement(self, ros, wall, *, xy, value, sample_count):
        if self.active is None:
            return
        if not all(math.isfinite(v) for v in (*xy, value)):
            self.event('measurement_failed', ros, wall, reason='nonfinite measurement')
            return
        run = self.active
        if run['first_ros'] is None:
            run['first_ros'], run['first_wall'] = ros, wall
        run['measurement_count'] += 1
        run['last_measurement'] = (float(xy[0]), float(xy[1]), float(value))
        self.event('measurement_complete', ros, wall, xy=xy, value=value,
                   sample_count=sample_count, measurement_count=run['measurement_count'])

    def finish(self, ros, wall, *, outcome, reason, robot_xy, source_end):
        if self.active is None:
            return None
        self._clock(ros, wall)
        run = self.active
        first = run['first_ros'] is not None
        valid = run['clock_valid']
        def durations(start, split, end):
            return ((split-start, end-split, end-start) if split is not None
                    else (end-start, None, end-start))
        initial, search, total = durations(run['start_ros'], run['first_ros'], ros)
        wi, ws, wt = durations(run['start_wall'], run['first_wall'], wall)
        measured = run['last_measurement']
        # A failed/interrupted trial does not claim its last pose is a source.
        estimated = measured[:2] if outcome == 'detected' and measured is not None else None
        error = math.dist(estimated, source_end) if estimated is not None and source_end is not None else None
        def coord(xy, index):
            return None if xy is None else xy[index]
        row = dict(
            run_id=run['run_id'], event_id=run['event_id'], lrs_lap=run['lrs_lap'],
            start_utc=run['start_utc'], time_basis=self.time_basis,
            initial_seconds=initial if valid else None,
            search_seconds=search if valid else None,
            total_seconds=total if valid else None,
            wall_initial_seconds=wi, wall_search_seconds=ws, wall_total_seconds=wt,
            initial_measurement_completed=first, ros_clock_valid=valid,
            source_start_x=coord(run['source_start'], 0), source_start_y=coord(run['source_start'], 1),
            source_x=coord(source_end, 0), source_y=coord(source_end, 1),
            source_position_available=source_end is not None,
            estimated_source_x=coord(estimated, 0), estimated_source_y=coord(estimated, 1),
            final_robot_x=coord(robot_xy, 0), final_robot_y=coord(robot_xy, 1),
            last_measurement_x=coord(measured, 0), last_measurement_y=coord(measured, 1),
            final_measured_concentration=coord(measured, 2),
            measurement_count=run['measurement_count'], outcome=outcome,
            gas_source_detected=outcome == 'detected',
            gas_source_measurement_failed=None if outcome == 'interrupted' else outcome == 'failed',
            termination_reason=reason, source_position_error=error,
            parameters=json.dumps(run['parameters'], ensure_ascii=False, sort_keys=True))
        self.event('hrs_end', ros, wall, outcome=outcome, reason=reason, source=source_end)
        self._append('runs.csv', row)
        self.active = None
        return row
