"""Evaluation references, immutable domains, clocks and score aggregation."""
import csv
import json
import math

import numpy as np
import pytest

from icir_cleanroom.gas_mapping.application.benchmark import EvaluationDomain, cost, score
from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.mapping.domains import sampling_mask, navigation_goal_mask


def domain(mask=None):
    grid = GridGeometry(3, 3, .5, -.25, -.25)
    return EvaluationDomain.snapshot(grid, np.ones((3, 3), bool) if mask is None else mask, grid)


def test_reachable_source_is_preserved_even_between_cell_centers():
    reference = domain().reference((.1, .1))
    row = reference.evaluate((.5, .5), (.1, .1))
    assert reference.candidates == ((.1, .1),)
    assert row['adjusted_source_position_error'] == pytest.approx(math.sqrt(.32))
    assert row['evaluation_reference_adjusted'] is False


def test_all_equidistant_cells_count_and_domain_is_frozen():
    mask = np.ones((3, 3), bool)
    mask[1, 1] = False
    frozen = domain(mask)
    mask[:] = False
    reference = frozen.reference((.5, .5))
    assert len(reference.candidates) == 4
    assert reference.distance == .5
    for xy in reference.candidates:
        assert reference.evaluate(xy, (.5, .5))['adjusted_source_position_error'] == 0
    assert reference.evaluate((1., 1.), (.5, .5))['adjusted_source_position_error'] == .5


@pytest.mark.parametrize('source,estimate,reason', [
    (None, (0, 0), 'missing_source'), ((0, 0), None, 'missing_estimate'),
    ((float('nan'), 0), (0, 0), 'missing_source')])
def test_unavailable_inputs(source, estimate, reason):
    row = domain().reference(source).evaluate(estimate, source)
    assert row['evaluation_available'] is False
    assert row['evaluation_reason'] == reason
    assert row['adjusted_source_position_error'] is None


def test_empty_domain_and_changed_source_are_not_raw_error_fallbacks():
    ref = domain(np.zeros((3, 3), bool)).reference((0, 0))
    assert ref.evaluate((0, 0), (0, 0))['evaluation_reason'] == 'no_reachable_cells'
    assert domain().reference((0, 0)).evaluate((0, 0), (.5, 0))['evaluation_reason'] == 'source_changed'
    assert EvaluationDomain.snapshot(None, None, None) is None


@pytest.mark.parametrize('blocked', [100, -1])
def test_obstacles_unknown_and_disconnected_space_share_navigation_rules(map_message_factory, blocked):
    cells = np.zeros((7, 9), dtype=int)
    cells[:, 4] = blocked
    msg = map_message_factory(cells.ravel(), width=9, height=7, resolution=.5)
    traversal = sampling_mask(msg, .5, 1.25, 1.25)
    goals = navigation_goal_mask(msg, traversal, .5)
    grid = GridGeometry.from_message(msg)
    ref = EvaluationDomain.snapshot(grid, goals, grid).reference((3.25, 1.25))
    assert ref.adjusted and all(x < 2 for x, _ in ref.candidates)
    for x, y in ref.candidates:
        row, col = grid.world_to_cell(x, y)
        assert goals[row, col]


def test_score_averages_cost_before_transform_and_is_independent_of_other_methods():
    costs = [cost(80, 0), cost(320, 1)]
    expected = 100/(1+sum(costs)/2)
    assert score(costs) == expected
    assert score(costs) != sum(score([c]) for c in costs)/2
    cost(10000, 50)
    assert score(costs) == expected
    for values in [[], [-1], [float('nan')]]:
        with pytest.raises(ValueError):
            score(values)


def test_multilap_timing_includes_gaps_and_resets_after_finish(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start_lrs(10, 100, event_id='e', lrs_lap=1, method_id='M1', evaluation_domain=domain())
    a = log.finish_lrs(20, 115, outcome='completed', reason='lap')
    log.start_lrs(25, 125, event_id='e', lrs_lap=2, method_id='M1')
    b = log.finish_lrs(35, 140, outcome='completed', reason='lap')
    log.start(40, 150, event_id='e', lrs_lap=2, source=(0, 0), parameters={'method':'M1'})
    log.measurement(45, 155, xy=(.5, 0), value=1, sample_count=1, source=(0,0))
    r = log.finish(50, 170, reason='done', robot_xy=(.5,0), source_end=(0,0))
    assert (a['lrs_seconds'], b['cumulative_lrs_seconds']) == (10, 20)
    assert (r['total_seconds'], r['lrs_seconds'], r['cumulative_lrs_seconds']) == (10,10,20)
    assert (r['experiment_seconds'], r['wall_experiment_seconds']) == (40,70)
    assert r['wall_cumulative_lrs_seconds'] == 30
    assert r['adjusted_source_position_error'] == .5
    history = list(csv.DictReader((log.directory/'estimate_history.csv').open()))
    assert float(history[-1]['adjusted_source_position_error']) == r['adjusted_source_position_error']
    log.start_lrs(60, 180, event_id='next', lrs_lap=3, method_id='M1')
    interrupted = log.finish_lrs(62, 185, outcome='interrupted', reason='node_shutdown')
    assert interrupted['experiment_seconds'] == 2 and not interrupted['experiment_completed']


def test_clock_reversal_in_lrs_invalidates_experiment_but_not_hrs(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start_lrs(10, 100, event_id='e', lrs_lap=1, method_id='M1')
    log.observe_time(5, 105)
    lap = log.finish_lrs(20, 110, outcome='completed', reason='lap')
    assert lap['lrs_seconds'] is None
    log.start(20, 110, event_id='e', lrs_lap=1, source=(0,0), parameters={'method':'M1'})
    row = log.finish(30, 120, reason='node_shutdown', robot_xy=None, source_end=(0,0))
    assert row['total_seconds'] == 10 and row['experiment_seconds'] is None
    assert row['wall_experiment_seconds'] == 20
    assert row['adjusted_source_position_error'] is None
    assert not row['experiment_completed']


def test_domain_at_first_lrs_is_used_even_if_live_map_changes(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    mask = np.ones((3, 3), bool); mask[1,1] = False
    log.start_lrs(0, 0, event_id='e', lrs_lap=1, method_id='M1', evaluation_domain=domain(mask))
    log.finish_lrs(1, 1, outcome='completed', reason='lap')
    log.start(1, 1, event_id='e', lrs_lap=1, source=(.5,.5), parameters={'method':'M1'}, evaluation_domain=domain())
    log.measurement(2, 2, xy=(0,.5), value=1, sample_count=1, source=(.5,.5))
    row = log.finish(3, 3, reason='done', robot_xy=(0,.5), source_end=(.5,.5))
    assert row['source_position_error'] == .5
    assert row['adjusted_source_position_error'] == 0
    assert len(json.loads(row['evaluation_reference_candidates'])) == 4


def test_hrs_clock_reversal_between_log_events_is_detected(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start_lrs(0, 0, event_id='e', lrs_lap=1, method_id='M1', evaluation_domain=domain())
    log.finish_lrs(10, 10, outcome='completed', reason='lap')
    log.start(15, 15, event_id='e', lrs_lap=1, source=(0, 0), parameters={'method':'M1'})
    log.observe_time(20, 20)
    log.observe_time(18, 21)  # Pose callback observes rewind before the next log event.
    row = log.finish(25, 30, reason='node_shutdown', robot_xy=None, source_end=(0,0))
    assert row['total_seconds'] is None and not row['ros_clock_valid']
    assert row['experiment_seconds'] is None
    assert row['wall_total_seconds'] == 15 and row['wall_experiment_seconds'] == 30


def test_wall_clock_reversal_invalidates_wall_experiment_time(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start_lrs(0, 10, event_id='e', lrs_lap=1, method_id='M1')
    log.observe_time(1, 9)
    row = log.finish_lrs(2, 12, outcome='interrupted', reason='node_shutdown')
    assert row['experiment_seconds'] == 2
    assert row['wall_lrs_seconds'] is None
    assert row['wall_experiment_seconds'] is None


def test_free_source_inside_safety_margin_is_projected(map_message_factory):
    cells = np.zeros((7, 9), dtype=int)
    cells[:, 4] = 100
    msg = map_message_factory(cells.ravel(), width=9, height=7, resolution=.5)
    traversal = sampling_mask(msg, .5, 1.25, 1.25)
    goals = navigation_goal_mask(msg, traversal, .5)
    grid = GridGeometry.from_message(msg)
    source = grid.cell_center(2, 3)
    assert cells[2, 3] == 0 and traversal[2, 3] and not goals[2, 3]
    ref = EvaluationDomain.snapshot(grid, goals, grid).reference(source)
    assert ref.adjusted and ref.candidates == (grid.cell_center(2, 2),)


def test_reference_tie_uses_absolute_meter_tolerance():
    mask = np.zeros((3, 3), bool)
    mask[1, 0] = mask[1, 2] = True
    d = domain(mask)
    assert len(d.reference((.5+2e-10, .5)).candidates) == 2
    assert d.reference((.5+2e-9, .5)).candidates == ((1., .5),)


def test_missing_map_keeps_raw_error_but_has_no_adjusted_error(tmp_path):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start(0, 0, event_id='e', lrs_lap=1, source=(0,0), parameters={'method':'M1'})
    log.measurement(1, 1, xy=(.5,0), value=1, sample_count=1, source=(0,0))
    row = log.finish(2, 2, reason='done', robot_xy=(.5,0), source_end=(0,0))
    assert row['source_position_error'] == .5
    assert row['adjusted_source_position_error'] is None
    assert row['evaluation_reason'] == 'missing_map'
