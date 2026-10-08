"""Time/error records must observe the search without changing its estimate."""
import csv
from types import SimpleNamespace as NS

import pytest

from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.ros.hrs_run_logging import record_hrs


def rows(log):
    with (log.directory / 'estimate_history.csv').open() as stream:
        return list(csv.DictReader(stream))


def start(log, method='M1', source=(0., 0.), time=100.):
    log.start(time, time+1000., event_id='source-1', lrs_lap=1,
              source=source, parameters={'method': method}, robot_xy=(0., 0.))


@pytest.mark.parametrize('method', [f'M{i}' for i in range(1, 8)])
def test_history_tracks_best_concentration_not_last_or_closest_pose(tmp_path, method):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    start(log, method)
    snap = lambda x, y: (round(x), round(y))
    for time, xy, value in [(105., (3.1, 4.1), .4),
                            (110., (0., 0.), .2),
                            (115., (6.1, 8.1), .8)]:
        log.position(xy)
        log.measurement(time, time+1000., xy=xy, value=value, sample_count=10,
                        source=(0., 0.), to_cell_center=snap)
        assert log.active['final_estimate'] is None  # Recording must not freeze it.

    history = rows(log)
    assert [r['elapsed_seconds'] for r in history] == ['0.0', '5.0', '10.0', '15.0']
    assert [r['source_position_error'] for r in history] == ['', '5.0', '5.0', '10.0']
    assert all(r['method_id'] == method for r in history)
    assert len({r['run_id'] for r in history}) == 1
    assert history[0]['measurement_count'] == '0'
    assert history[0]['estimated_source_x'] == ''
    assert float(history[-1]['hrs_distance_m']) == pytest.approx(log.active['distance_m'])

    # Final movement increases T/D without changing the estimated source.
    log.final_estimate(snap)
    log.position((9., 8.))
    final = log.finish(130., 1130., reason='done', robot_xy=(6., 8.), source_end=(0., 0.))
    end = rows(log)[-1]
    assert end['event'] == 'hrs_end'
    for field, key in [('elapsed_seconds', 'total_seconds'),
                       ('hrs_distance_m', 'hrs_distance_m'),
                       ('source_position_error', 'source_position_error')]:
        assert float(end[field]) == pytest.approx(final[key])
    assert float(end['elapsed_seconds']) == 30.
    assert float(end['source_position_error']) == 10.
    assert {'outcome', 'gas_source_detected', 'gas_source_measurement_failed',
            'score', 'normalized_error'}.isdisjoint(end)


def test_missing_truth_and_invalid_clock_do_not_invent_values(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    start(log)
    log.measurement(105., 1105., xy=(3., 4.), value=.5, sample_count=10, source=None)
    assert rows(log)[-1]['source_position_error'] == ''
    assert rows(log)[-1]['source_x'] == ''  # Do not substitute stale start truth.
    log.measurement(106., 1106., xy=(float('nan'), 0.), value=.6, sample_count=10)
    assert len(rows(log)) == 2
    log.finish(1., 1110., reason='node_shutdown', robot_xy=(3.,4.), source_end=None)
    end = rows(log)[-1]
    assert end['elapsed_seconds'] == '' and end['ros_clock_valid'] == 'False'
    assert end['wall_elapsed_seconds'] == '10.0'
    assert end['source_position_error'] == ''
    assert log.finish(2., 1111., reason='duplicate', robot_xy=None, source_end=None) is None
    assert len(rows(log)) == 3

    start(log, time=200.)
    first = rows(log)[-1]
    assert first['run_id'] != end['run_id']
    assert first['elapsed_seconds'] == '0.0' and first['source_position_error'] == ''


def test_ros_measurement_and_final_estimate_use_same_grid_and_current_truth(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    start(log)
    now = [105.]
    controller = NS(
        hrs_run_log=log, hrs_log_source=(2., 4.), latest_pose=None,
        gmrf=NS(geometry=NS(world_to_cell=lambda x,y:(0,0),
                           cell_center=lambda r,c:(2.,3.))),
        get_clock=lambda: NS(now=lambda: NS(nanoseconds=int(now[0]*1e9))),
        get_logger=lambda: NS(info=lambda msg: None, error=lambda msg: pytest.fail(msg)))
    record_hrs(controller, 'measurement', xy=(2.1,3.4), value=.8, sample_count=10)
    point = rows(log)[-1]
    assert point['estimated_source_x'] == '2.0'
    assert point['estimated_source_y'] == '3.0'
    assert point['source_position_error'] == '1.0'
    assert log.active['final_estimate'] is None
    now[0] = 110.
    estimate = record_hrs(controller, 'estimate')
    assert rows(log)[-1]['event'] == 'final_estimate_selected'
    assert estimate['estimated_source_x'] == float(point['estimated_source_x'])
    controller.hrs_log_source = None
    now[0] = 120.
    result = record_hrs(controller, 'finish', reason='done')
    assert result['source_position_error'] is None
    assert rows(log)[-1]['source_position_error'] == ''
