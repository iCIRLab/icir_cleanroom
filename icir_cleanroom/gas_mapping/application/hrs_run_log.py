"""LRS travel and HRS time, distance, and estimate records.

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

from .hrs_methods import METHODS
from .benchmark import SOURCE_TRUTH_FIELDS, EvaluationReference


def source_truth_fields(truth):
    """Blank fields when the environment published no truth for this run."""
    values = truth if isinstance(truth, dict) else {}
    return {name: values.get(name) for name in SOURCE_TRUTH_FIELDS}


class HrsRunLog:
    def __init__(self, directory, time_basis):
        self.directory = Path(directory).expanduser() / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8])
        self.time_basis = time_basis
        self.active = None
        self.active_lrs = None
        self.last_lrs = None
        self.run_count = 0
        self.lrs_run_count = 0
        self.episode = None

    def observe_time(self, ros, wall):
        """Observe clocks during LRS and transitions as well as HRS."""
        if self.active is not None:
            run = self.active
            if (not math.isfinite(ros) or not math.isfinite(wall)
                    or ros < run['last_ros'] or wall < run['last_wall']):
                run['clock_valid'] = False
            run['last_ros'], run['last_wall'] = ros, wall
        for clock in (self.episode, self.active_lrs):
            if clock is None:
                continue
            if not math.isfinite(ros) or ros < clock['last_ros']:
                clock['ros_valid'] = False
            if not math.isfinite(wall) or wall < clock['last_wall']:
                clock['wall_valid'] = False
            clock['last_ros'], clock['last_wall'] = ros, wall

    def _time_fields(self, ros, wall, completed=False):
        episode = self.episode
        if episode is None:
            return dict(lrs_seconds=None, cumulative_lrs_seconds=None,
                        experiment_seconds=None, wall_lrs_seconds=None,
                        wall_cumulative_lrs_seconds=None, wall_experiment_seconds=None,
                        experiment_start_ros_seconds=None, experiment_ros_clock_valid=False,
                        experiment_completed=completed)
        self.observe_time(ros, wall)
        lap = self.last_lrs if self.last_lrs and self.last_lrs['event_id'] == episode['event_id'] else {}
        return dict(
            lrs_seconds=lap.get('lrs_seconds'),
            cumulative_lrs_seconds=episode['lrs_seconds'] if episode['ros_valid'] else None,
            experiment_seconds=ros-episode['start_ros'] if episode['ros_valid'] else None,
            wall_lrs_seconds=lap.get('wall_total_seconds'),
            wall_cumulative_lrs_seconds=episode['wall_lrs_seconds'] if episode['wall_valid'] else None,
            wall_experiment_seconds=wall-episode['start_wall'] if episode['wall_valid'] else None,
            experiment_start_ros_seconds=episode['start_ros'],
            experiment_ros_clock_valid=episode['ros_valid'], experiment_completed=completed)

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
        self.observe_time(ros, wall)
        run = self.active
        if ros < run['last_ros'] or wall < run['last_wall']:
            run['clock_valid'] = False
        run['last_ros'], run['last_wall'] = ros, wall

    def event(self, kind, ros, wall, **details):
        if self.active is None:
            return
        self._clock(ros, wall)
        run = self.active
        if kind == 'target_selected':
            run['attempt_count'] += 1
            if run['stage'] == 'SEARCH':
                run['search_iterations'] += 1
        if kind == 'navigation_attempt':
            run['navigation_attempts'] += 1
        self._append('events.csv', dict(
            schema_version=5, method_id=run['method_id'],
            initial_selection=run['initial_selection'], search_method=run['search_method'],
            stage=run['stage'],
            run_id=run['run_id'], event=kind, ros_seconds=ros,
            wall_elapsed_seconds=wall-run['start_wall'],
            utc=datetime.now(timezone.utc).isoformat(),
            details=json.dumps(details, ensure_ascii=False, allow_nan=False)))

    def start(self, ros, wall, *, event_id, lrs_lap, source, parameters, robot_xy=None,
              evaluation_domain=None):
        if self.active is not None or self.active_lrs is not None:
            raise RuntimeError('previous LRS/HRS record is still active')
        self.run_count += 1
        if self.episode is not None and (self.episode['event_id'] != event_id or self.episode['closed']):
            self.episode = None
        self.observe_time(ros, wall)
        domain = self.episode['domain'] if self.episode is not None else evaluation_domain
        reference = (domain.reference(source) if domain is not None else
                     EvaluationReference(source=source, reason='missing_map'))
        method_id = parameters.get('method', 'M3')
        method = METHODS[method_id]
        self.active = dict(
            run_id=f'{self.directory.name}_{self.run_count:04d}',
            event_id=event_id, lrs_lap=lrs_lap, source_start=source,
            method_id=method_id, initial_selection=method.initial, search_method=method.search,
            stage='INITIAL', attempt_count=0, search_iterations=0, navigation_attempts=0,
            parameters=parameters, start_ros=ros, start_wall=wall,
            last_ros=ros, last_wall=wall, first_ros=None, first_wall=None,
            last_measurement=None, best_measurement=None, final_estimate=None,
            measurement_count=0, clock_valid=True, evaluation_reference=reference,
            last_robot_xy=None, distance_m=None,
            lrs_run_id=None, lrs_distance_m=None,
            start_utc=datetime.now(timezone.utc).isoformat())
        if (self.last_lrs is not None and
                self.last_lrs['event_id'] == event_id and
                self.last_lrs['lrs_lap'] == lrs_lap and
                self.last_lrs['outcome'] == 'completed'):
            self.active['lrs_run_id'] = self.last_lrs['lrs_run_id']
            self.active['lrs_distance_m'] = self.last_lrs['lrs_distance_m']
        self.position(robot_xy)
        self.event('lrs_complete_hrs_start', ros, wall,
                   source=source, parameters=parameters)
        self.record_estimate(ros, wall, event='hrs_start', source=source)

    def position(self, xy):
        """Accumulate sampled travel in the active LRS or HRS interval only."""
        run = self.active if self.active is not None else self.active_lrs
        if run is None or xy is None:
            return
        xy = (float(xy[0]), float(xy[1]))
        if not all(math.isfinite(value) for value in xy):
            return
        previous = run['last_robot_xy']
        if previous is None:
            run['distance_m'] = 0.0
        else:
            run['distance_m'] += math.dist(previous, xy)
        run['last_robot_xy'] = xy

    def start_lrs(self, ros, wall, *, event_id, lrs_lap, method_id, robot_xy=None,
                  evaluation_domain=None):
        if self.active is not None or self.active_lrs is not None:
            raise RuntimeError('previous LRS/HRS record is still active')
        self.lrs_run_count += 1
        if self.episode is None or self.episode['event_id'] != event_id or self.episode['closed']:
            self.episode = dict(event_id=event_id, start_ros=ros, start_wall=wall,
                                last_ros=ros, last_wall=wall, ros_valid=True, wall_valid=True,
                                lrs_seconds=0.0, wall_lrs_seconds=0.0, closed=False,
                                domain=evaluation_domain)
        self.observe_time(ros, wall)
        self.last_lrs = None
        self.active_lrs = dict(
            lrs_run_id=f'{self.directory.name}_lrs_{self.lrs_run_count:04d}',
            event_id=event_id, lrs_lap=lrs_lap, method_id=method_id,
            start_utc=datetime.now(timezone.utc).isoformat(),
            start_ros=ros, start_wall=wall, last_ros=ros, last_wall=wall,
            ros_valid=True, wall_valid=True,
            last_robot_xy=None, distance_m=None)
        self.position(robot_xy)

    def finish_lrs(self, ros, wall, *, outcome, reason, robot_xy=None,
                   hazard_detected=False):
        if self.active_lrs is None:
            return None
        self.position(robot_xy)
        run = self.active_lrs
        self.observe_time(ros, wall)
        elapsed = ros-run['start_ros']
        wall_elapsed = wall-run['start_wall']
        if run['ros_valid']:
            self.episode['lrs_seconds'] += elapsed
        if run['wall_valid']:
            self.episode['wall_lrs_seconds'] += wall_elapsed
        row = dict(
            schema_version=2, lrs_run_id=run['lrs_run_id'],
            event_id=run['event_id'], lrs_lap=run['lrs_lap'],
            method_id=run['method_id'], start_utc=run['start_utc'],
            time_basis=self.time_basis, start_ros_seconds=run['start_ros'],
            end_ros_seconds=ros, wall_total_seconds=wall_elapsed if run['wall_valid'] else None,
            lrs_seconds=elapsed if run['ros_valid'] else None, ros_clock_valid=run['ros_valid'],
            lrs_distance_m=run['distance_m'],
            hazard_detected=hazard_detected, outcome=outcome,
            termination_reason=reason)
        self.last_lrs = row
        row.update(self._time_fields(ros, wall))
        self._append('lrs_runs.csv', row)
        self.active_lrs = None
        if outcome != 'completed':
            self.episode['closed'] = True
        return row

    def lrs_measurement(self, ros, wall, *, xy, value, sample_count, cell,
                        patrol_index=None, hazard_threshold=None,
                        to_cell_center=None):
        """One patrol point: where the robot stood and what it measured."""
        if self.active_lrs is None:
            return None
        run = self.active_lrs
        self.observe_time(ros, wall)
        run['measurement_count'] = run.get('measurement_count', 0) + 1
        center = (None, None)
        if to_cell_center is not None and all(math.isfinite(v) for v in xy):
            center = to_cell_center(float(xy[0]), float(xy[1]))
        hazard = (None if hazard_threshold is None or not math.isfinite(value)
                  else bool(value >= float(hazard_threshold)))
        row = dict(
            schema_version=1, lrs_run_id=run['lrs_run_id'],
            event_id=run['event_id'], lrs_lap=run['lrs_lap'],
            method_id=run['method_id'], time_basis=self.time_basis,
            measurement_index=run['measurement_count'],
            patrol_index=patrol_index,
            utc=datetime.now(timezone.utc).isoformat(),
            ros_seconds=ros, lrs_elapsed_seconds=ros-run['start_ros'],
            wall_elapsed_seconds=wall-run['start_wall'],
            robot_x=float(xy[0]), robot_y=float(xy[1]),
            cell_row=None if cell is None else int(cell[0]),
            cell_col=None if cell is None else int(cell[1]),
            cell_center_x=center[0], cell_center_y=center[1],
            concentration=float(value), sample_count=sample_count,
            hazard_threshold=hazard_threshold, hazard=hazard,
            lrs_distance_m=run['distance_m'])
        self._append('lrs_measurements.csv', row)
        return row

    def measurement(self, ros, wall, *, xy, value, sample_count,
                    source=None, to_cell_center=None):
        if self.active is None:
            return
        if not all(math.isfinite(v) for v in (*xy, value)):
            self.event('measurement_failed', ros, wall, reason='nonfinite measurement')
            return
        run = self.active
        run['measurement_count'] += 1
        run['last_measurement'] = (float(xy[0]), float(xy[1]), float(value))
        if (run['best_measurement'] is None or
                float(value) > run['best_measurement'][2]):
            run['best_measurement'] = run['last_measurement']
        self.event('measurement_complete', ros, wall, xy=xy, value=value,
                   sample_count=sample_count, measurement_count=run['measurement_count'])
        self.record_estimate(ros, wall, event='measurement_complete',
                             source=source, to_cell_center=to_cell_center)

    def initial_complete(self, ros, wall):
        if self.active is None or self.active['first_ros'] is not None:
            return
        run = self.active
        if run['measurement_count'] == 0:
            raise ValueError('initial stage needs a valid measurement')
        run['first_ros'], run['first_wall'] = ros, wall
        self.event('initial_gmrf_complete', ros, wall)
        run['stage'] = 'SEARCH'

    def current_estimate(self, to_cell_center=None):
        """Return the best estimate so far without freezing the search result."""
        if self.active is None:
            return None
        run = self.active
        if run['final_estimate'] is not None:
            return dict(run['final_estimate'])
        best = run['best_measurement']
        raw_xy = None if best is None else best[:2]
        cell_xy = (to_cell_center(*raw_xy)
                   if raw_xy is not None and to_cell_center is not None else None)
        estimated = cell_xy if cell_xy is not None else raw_xy
        return dict(
            estimated_source_x=None if estimated is None else estimated[0],
            estimated_source_y=None if estimated is None else estimated[1],
            estimated_source_cell_x=None if cell_xy is None else cell_xy[0],
            estimated_source_cell_y=None if cell_xy is None else cell_xy[1])

    def record_estimate(self, ros, wall, *, event, source, to_cell_center=None):
        """Write event-time samples for E(t); truth never selects an estimate."""
        if self.active is None:
            return
        self._clock(ros, wall)
        run = self.active
        estimate = self.current_estimate(to_cell_center)
        xy = (None if estimate['estimated_source_x'] is None else
              (estimate['estimated_source_x'], estimate['estimated_source_y']))
        best = run['best_measurement']
        self._append('estimate_history.csv', dict(
            schema_version=2, run_id=run['run_id'], event_id=run['event_id'],
            method_id=run['method_id'], event=event, stage=run['stage'],
            time_basis=self.time_basis, ros_seconds=ros,
            elapsed_seconds=ros-run['start_ros'] if run['clock_valid'] else None,
            wall_elapsed_seconds=wall-run['start_wall'],
            ros_clock_valid=run['clock_valid'], hrs_distance_m=run['distance_m'],
            **estimate,
            source_x=None if source is None else source[0],
            source_y=None if source is None else source[1],
            source_position_error=(math.dist(xy, source)
                                   if xy is not None and source is not None else None),
            **run['evaluation_reference'].evaluate(xy, source),
            best_measured_concentration=None if best is None else best[2],
            measurement_count=run['measurement_count']))

    def final_estimate(self, to_cell_center=None):
        """Freeze the highest-measurement cell for marker, final goal, and E.

        The original collection position remains in best_measurement_x/y. Without
        grid geometry, the measured position itself is the estimate.
        """
        if self.active is None:
            return None
        run = self.active
        if run['final_estimate'] is None:
            run['final_estimate'] = self.current_estimate(to_cell_center)
        return dict(run['final_estimate'])

    def finish(self, ros, wall, *, reason, robot_xy, source_end,
               to_cell_center=None, source_truth=None):
        if self.active is None:
            return None
        self.position(robot_xy)
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
        best = run['best_measurement']
        estimate = self.final_estimate(to_cell_center)
        estimated = (None if estimate['estimated_source_x'] is None else
                     (estimate['estimated_source_x'], estimate['estimated_source_y']))
        error = math.dist(estimated, source_end) if estimated is not None and source_end is not None else None
        def coord(xy, index):
            return None if xy is None else xy[index]
        row = dict(
            schema_version=6, start_boundary='lrs_complete_hrs_start',
            end_boundary='hrs_end',
            initial_boundary='first_valid_measurement_gmrf_complete',
            method_id=run['method_id'], initial_selection=run['initial_selection'],
            search_method=run['search_method'], final_stage=run['stage'],
            attempt_count=run['attempt_count'], search_iterations=run['search_iterations'],
            navigation_attempts=run['navigation_attempts'],            run_id=run['run_id'], event_id=run['event_id'], lrs_lap=run['lrs_lap'],
            start_utc=run['start_utc'], time_basis=self.time_basis,
            initial_seconds=initial if valid else None,
            search_seconds=search if valid else None,
            total_seconds=total if valid else None,
            total_distance_m=(run['lrs_distance_m'] + run['distance_m']
                              if run['lrs_distance_m'] is not None and
                              run['distance_m'] is not None else None),
            hrs_distance_m=run['distance_m'],
            lrs_distance_m=run['lrs_distance_m'], lrs_run_id=run['lrs_run_id'],
            wall_initial_seconds=wi, wall_search_seconds=ws, wall_total_seconds=wt,
            initial_measurement_completed=run['measurement_count'] > 0,
            initial_update_completed=first, ros_clock_valid=valid,
            source_start_x=coord(run['source_start'], 0), source_start_y=coord(run['source_start'], 1),
            source_x=coord(source_end, 0), source_y=coord(source_end, 1),
            source_position_available=source_end is not None,
            **source_truth_fields(source_truth),
            **estimate,
            final_robot_x=coord(robot_xy, 0), final_robot_y=coord(robot_xy, 1),
            last_measurement_x=coord(measured, 0), last_measurement_y=coord(measured, 1),
            final_measured_concentration=coord(measured, 2),
            best_measurement_x=coord(best, 0), best_measurement_y=coord(best, 1),
            best_measured_concentration=coord(best, 2),
            measurement_count=run['measurement_count'],
            termination_reason=reason, source_position_error=error,
            **run['evaluation_reference'].evaluate(estimated, source_end),
            **self._time_fields(ros, wall, completed=(
                reason != 'node_shutdown' and 'final approach stopped:' not in reason
                and estimated is not None)),
            parameters=json.dumps(run['parameters'], ensure_ascii=False, sort_keys=True))
        self.event('hrs_end', ros, wall, reason=reason, source=source_end)
        self.record_estimate(ros, wall, event='hrs_end', source=source_end)
        self._append('runs.csv', row)
        self.active = None
        if self.episode is not None:
            self.episode['closed'] = True
        return row
