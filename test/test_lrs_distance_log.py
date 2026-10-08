"""Travel boundaries across LRS laps, HRS, and shutdown."""
import csv
from types import SimpleNamespace as NS

import pytest

from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.config import ControllerConfig
from icir_cleanroom.gas_mapping.ros.controller_node import GasMappingControllerNode
from icir_cleanroom.gas_mapping.ros.hrs_run_logging import record_hrs, record_lrs
from icir_cleanroom.gas_mapping.ros.lrs_workflow import LrsWorkflow


def read_rows(log, filename):
    with (log.directory / filename).open() as stream:
        return list(csv.DictReader(stream))


def start_lrs(log, lap=1, xy=(0., 0.)):
    log.start_lrs(10., 100., event_id='event-1', lrs_lap=lap,
                  method_id='M3', robot_xy=xy)


def test_every_lrs_lap_is_saved_and_hrs_links_only_the_preceding_lap(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.position((99., 99.))  # Waiting before patrol is excluded.
    start_lrs(log)
    log.position((3., 4.))
    first = log.finish_lrs(20., 110., outcome='completed', reason='lap',
                           robot_xy=(0., 0.))  # Includes the return leg.
    assert first['lrs_distance_m'] == pytest.approx(10.)
    assert not (log.directory / 'runs.csv').exists()
    start_lrs(log, lap=2)
    log.position((0., 2.))
    second = log.finish_lrs(30., 120., outcome='completed', reason='lap',
                            robot_xy=(0., 0.), hazard_detected=True)
    assert second['lrs_distance_m'] == pytest.approx(4.)
    log.start(30., 120., event_id='event-1', lrs_lap=2,
              source=None, parameters={}, robot_xy=(0., 0.))
    log.position((3., 4.))  # First HRS candidate approach.
    row = log.finish(40., 130., reason='done',
                     robot_xy=(6., 4.), source_end=None)
    assert row['lrs_distance_m'] == pytest.approx(4.)
    assert row['hrs_distance_m'] == pytest.approx(8.)
    assert row['total_distance_m'] == pytest.approx(12.)
    assert row['schema_version'] == 6
    assert row['lrs_run_id'] == second['lrs_run_id']
    log.position((99., 99.))  # Travel after HRS completion is excluded.
    assert [float(r['lrs_distance_m']) for r in read_rows(log, 'lrs_runs.csv')] == [10., 4.]
    saved = read_rows(log, 'runs.csv')[0]
    assert (saved['lrs_distance_m'], saved['hrs_distance_m']) == ('4.0', '8.0')
    assert float(saved['total_distance_m']) == pytest.approx(12.)


@pytest.mark.parametrize('outcome,xy,expected', [
    ('interrupted', (3., 4.), 5.), ('completed', (0., 0.), 0.),
    ('interrupted', None, None)])
def test_lrs_partial_stationary_and_missing_pose_records(tmp_path, outcome, xy, expected):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    start_lrs(log, xy=None if xy is None else (0., 0.))
    log.position((float('nan'), 1.))
    row = log.finish_lrs(20., 110., outcome=outcome, reason='test', robot_xy=xy)
    assert row['lrs_distance_m'] == expected
    assert row['outcome'] == outcome
    assert log.finish_lrs(20., 110., outcome=outcome, reason='duplicate') is None
    assert len(read_rows(log, 'lrs_runs.csv')) == 1
    start_lrs(log, lap=2, xy=(50., 50.))
    assert log.active_lrs['distance_m'] == 0.


@pytest.mark.parametrize('event,lap', [('event-2', 1), ('event-1', 2)])
def test_unrelated_lrs_distance_is_not_reused(tmp_path, event, lap):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    start_lrs(log)
    log.finish_lrs(20., 110., outcome='completed', reason='lap', robot_xy=(3., 4.))
    log.start(30., 120., event_id=event, lrs_lap=lap, source=None, parameters={})
    row = log.finish(40., 130., reason='done', robot_xy=None, source_end=None)
    assert row['lrs_distance_m'] is None and row['lrs_run_id'] is None
    assert row['total_distance_m'] is None


def test_lrs_and_hrs_cannot_overlap(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    start_lrs(log)
    with pytest.raises(RuntimeError, match='still active'):
        log.start(20., 110., event_id='event-1', lrs_lap=1, source=None, parameters={})
    with pytest.raises(RuntimeError, match='still active'):
        start_lrs(log, lap=2)
    log.finish_lrs(20., 110., outcome='completed', reason='lap')
    log.start(20., 110., event_id='event-1', lrs_lap=1, source=None, parameters={})
    with pytest.raises(RuntimeError, match='still active'):
        start_lrs(log, lap=2)


@pytest.mark.parametrize('lrs_xy,hrs_xy,expected', [
    ((0., 0.), (0., 0.), 0.), (None, (0., 0.), None), ((0., 0.), None, None)])
def test_total_distance_requires_both_components_but_accepts_zero(tmp_path, lrs_xy, hrs_xy, expected):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    start_lrs(log, xy=lrs_xy)
    log.finish_lrs(20., 110., outcome='completed', reason='lap', robot_xy=lrs_xy)
    log.start(20., 110., event_id='event-1', lrs_lap=1, source=None,
              parameters={}, robot_xy=hrs_xy)
    row = log.finish(30., 120., reason='done',
                     robot_xy=hrs_xy, source_end=None)
    assert row['total_distance_m'] == expected
    assert read_rows(log, 'runs.csv')[0]['total_distance_m'] == ('' if expected is None else '0.0')


@pytest.mark.parametrize('hazard', [True, False])
def test_ros_lrs_finish_routes_pose_updates_to_the_next_mode(tmp_path, hazard):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    c = NS(hrs_run_log=log, hrs_log_source=None, current_event_id='event-1',
           lrs_lap=1, method='M3', config=ControllerConfig.defaults(),
           lap_hazard_detected=hazard, lap_max_concentration=.5, hazard_threshold=.2,
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=100_000_000_000)),
           get_logger=lambda: NS(info=lambda msg: None, error=lambda msg: pytest.fail(msg)),
           persist_history=lambda msg: None, commit_history_snapshot=lambda msg: None,
           publish_phase=lambda phase: None)
    def pose(x, y):
        GasMappingControllerNode.pose_callback(c, NS(pose=NS(position=NS(x=x, y=y))))
    def next_lrs(reason):
        assert log.active_lrs is None and log.active is None
        c.lrs_lap += 1
        record_lrs(c, 'start')
    def next_hrs():
        assert log.active_lrs is None
        assert log.active['lrs_distance_m'] == pytest.approx(10.)
    c.start_lrs_lap, c.start_hrs_planning = next_lrs, next_hrs
    pose(0., 0.)
    record_lrs(c, 'start')
    pose(3., 4.)
    pose(0., 0.)
    LrsWorkflow(c).finish_lrs_navigation()
    assert float(read_rows(log, 'lrs_runs.csv')[0]['lrs_distance_m']) == pytest.approx(10.)
    pose(0., 2.)
    if hazard:
        row = record_hrs(c, 'finish', reason='node_shutdown')
        assert row['lrs_distance_m'] == pytest.approx(10.)
        assert row['hrs_distance_m'] == pytest.approx(2.)
        assert row['total_distance_m'] == pytest.approx(12.)
    else:
        row = record_lrs(c, 'finish', outcome='interrupted', reason='node_shutdown')
        assert row['lrs_lap'] == 2 and row['lrs_distance_m'] == pytest.approx(2.)
        assert not (log.directory / 'runs.csv').exists()
