"""Offline strategy and ROS adapter checks; no simulator or navigation goals."""
import csv
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from icir_cleanroom.gas_mapping.application.hrs import HrsManager
from icir_cleanroom.gas_mapping.application.hrs_methods import (
    METHODS, HrsSession, SquareSpiral, lrs_max, lrs_centroid)
from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.mapping.gmrf import GmrfGrid
from icir_cleanroom.gas_mapping.mapping.path_distance import SamplingDistanceOracle
from icir_cleanroom.gas_mapping.models import GasMeasurement, Pose2D, MappingPhase
from icir_cleanroom.gas_mapping.config import ControllerConfig


def measurement(x, y, value):
    return GasMeasurement(Pose2D(x, y), value, 0, None, MappingPhase.LRS, 1., 10)


def setup_grid(factory):
    g = GmrfGrid(factory(width=7, height=7, resolution=.5))
    g.solution[:] = np.linspace(.1, .8, len(g.solution))
    g.variance[:] = .01
    oracle = SamplingDistanceOracle(g.geometry, np.ones((7, 7), bool))
    return g, oracle


def test_registry_and_defaults():
    assert [(m.initial, m.search) for m in METHODS.values()] == [
        ('lrs_max', 'dducb'), ('lrs_max', 'spiral'), ('dducb', 'dducb'),
        ('dducb', 'spiral'), ('lrs_centroid', 'dducb'),
        ('lrs_centroid', 'spiral'), ('ucb', 'ucb')]
    assert ControllerConfig.defaults().hrs.method == 'M3'


@pytest.mark.parametrize('key,value', [('method', 'M8'), ('spiral_step_cells', 0),
    ('spiral_step_cells', 1.5), ('spiral_recenter_epsilon', -1.),
    ('spiral_recenter_epsilon', float('nan')), ('hrs_search_patience', 0),
    ('hrs_search_patience', 1.5), ('hrs_search_min_relative_improvement', -0.01),
    ('hrs_search_min_relative_improvement', float('nan'))])
def test_invalid_method_settings(key, value):
    parameters = ControllerConfig.defaults().flat_values()
    parameters[key] = value
    with pytest.raises(ValueError):
        ControllerConfig.from_mapping(parameters)


def test_lrs_points_weights_ties_and_invalid_data():
    ms = [measurement(0., 0., 1.), measurement(4., 2., 1.), measurement(99., 99., -2.)]
    assert lrs_max(ms) == (0., 0.)
    assert lrs_centroid(ms) == (2., 1.)
    for ms in [[], [measurement(1., 2., 0.)]]:
        with pytest.raises(ValueError):
            lrs_centroid(ms)
    s = HrsSession('M1', [measurement(0., 0., float('nan'))])
    with pytest.raises(ValueError):
        lrs_max(s.measurements)


def test_lrs_target_always_snaps_to_nearest_cell_center(map_message_factory):
    g, o = setup_grid(map_message_factory)
    candidates = HrsManager().build_candidates(g, set(), .2)
    s = HrsSession('M1', [measurement(.3, .3, .8)])
    _, c = s.select(candidates, gmrf=g, current_xy=(.25, .25), oracle=o,
                    distance_weight=.03, position_valid=lambda x, y: True)
    assert (c.x, c.y) == (.25, .25)  # cell center, not the raw (.3, .3) request
    assert s.requested_xy == (.25, .25)  # requested_xy is itself cell-snapped now
    s = HrsSession('M5', [measurement(-10., -10., .8)])
    _, c = s.select(candidates, gmrf=g, current_xy=(.25, .25), oracle=o,
                    distance_weight=.03, position_valid=lambda x, y: False)
    assert (c.x, c.y) == (.25, .25)  # nearest eligible cell is unchanged
    assert s.requested_xy == (-9.75, -9.75)  # snapped to the cell containing (-10, -10)


def test_m3_scores_target_match_existing_and_m7_never_reads_distances(map_message_factory):
    g, o = setup_grid(map_message_factory)
    candidates = HrsManager().build_candidates(g, set(), .2)
    old, expected = HrsManager.select_candidate(candidates, (.25, .25), o.distance, .03)
    s = HrsSession('M3')
    new, selected = s.select(candidates, gmrf=g, current_xy=(.25, .25), oracle=o, distance_weight=.03)
    assert selected.variable == expected.variable
    assert [c.score for c in old] == pytest.approx([c.score for c in new])
    _, selected = HrsSession('M7').select(candidates, gmrf=g, current_xy=(.25, .25),
                                        oracle=None, distance_weight=999.)
    assert selected.ucb == max(c.ucb for c in candidates)


def test_spiral_order_recentering_epsilon_and_finite_coverage():
    s = SquareSpiral()
    s.recenter((2, 2), .5, (0, 4, 0, 4))
    cells = list(s.cursor)
    assert cells[:8] == [(2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3)]
    assert {(r, c) for r in range(5) for c in range(5)} - {(2, 2)} <= set(cells)
    assert len(cells) == len(set(cells))
    assert not s.observe((2, 3), .5, (0, 4, 0, 4))
    assert s.observe((3, 3), .501, (0, 4, 0, 4))
    assert s.center == (3, 3) and (2, 3) in s.visited
    s.epsilon = .1
    assert not s.observe((3, 4), .55, (0, 4, 0, 4))


def test_spiral_skips_blocked_visited_and_exhausts(map_message_factory):
    g, _ = setup_grid(map_message_factory)
    candidates = HrsManager().build_candidates(g, set(), .2)
    s = SquareSpiral()
    s.recenter((0, 0), .2, (0, 6, 0, 6))
    s.visited.add((0, 1))
    allowed = tuple(c for c in candidates if (c.row, c.col) == (2, 2))
    assert s.select(allowed)[1].row == 2
    assert s.select(allowed)[1] is None


def test_search_stall_uses_relative_improvement_and_resets_on_gain(map_message_factory):
    g, _ = setup_grid(map_message_factory)
    s = HrsSession('M3', patience=2, min_relative_improvement=0.1)
    s.updated(g.variable_at(0, 0), .5, g)  # INITIAL baseline; never counts toward the streak
    assert s.stage == 'SEARCH' and s.best_value == .5 and s.no_improvement_streak == 0
    assert not s.stalled()
    s.updated(g.variable_at(1, 1), .52, g)  # +4% < 10% required -> no improvement
    assert s.no_improvement_streak == 1 and not s.stalled()
    s.updated(g.variable_at(2, 2), .53, g)  # 2nd consecutive non-improvement -> patience reached
    assert s.no_improvement_streak == 2 and s.stalled()
    s.updated(g.variable_at(3, 3), .7, g)  # +40% >= 10% -> resets the streak
    assert s.best_value == .7 and s.no_improvement_streak == 0 and not s.stalled()


def test_search_stall_zero_baseline_requires_strict_gain(map_message_factory):
    g, _ = setup_grid(map_message_factory)
    s = HrsSession('M3', patience=1, min_relative_improvement=0.05)
    s.updated(g.variable_at(0, 0), 0., g)
    assert s.best_value == 0. and not s.stalled()
    s.updated(g.variable_at(1, 1), 0., g)  # not strictly greater than a zero baseline
    assert s.stalled()


@pytest.mark.parametrize('method', list(METHODS))
def test_all_methods_isolated_three_measurement_smoke(method, map_message_factory, tmp_path):
    # Recreate all mutable map, oracle, measurements and strategy state for each method.
    g, o = setup_grid(map_message_factory)
    ms = [measurement(.75, .75, .4), measurement(1.75, 1.75, .6)]
    s = HrsSession(method, ms)
    manager = HrsManager()
    visited = {g.variable_at(1, 1), g.variable_at(3, 3)}
    initial_visited = set(visited)
    xy = (.25, .25)
    log = HrsRunLog(tmp_path/method, 'offline_test_clock')
    log.start(0., 0., event_id='test', lrs_lap=1, source=None, parameters={'method': method})
    path = []
    for iteration in range(3):
        excluded = set() if iteration == 0 and s.method.initial.startswith('lrs_') else visited
        candidates = manager.build_candidates(g, excluded, .2)
        _, target = s.select(candidates, gmrf=g, current_xy=xy, oracle=o,
                             distance_weight=.03, position_valid=lambda x, y: True)
        assert target is not None
        if iteration > 0:
            assert target.variable not in visited
        log.event('target_selected', iteration*3., iteration*3.)
        xy = (target.x, target.y)
        path.append(target.variable)
        value = .3 + iteration*.1
        g.set_observation(*xy, value)
        # Real GMRF solver update, with deterministic test observations.
        g.solve_gabp()
        visited.add(target.variable)
        log.measurement(iteration*3.+1., iteration*3.+1., xy=xy, value=value, sample_count=10)
        s.updated(target.variable, value, g)
        log.initial_complete(iteration*3.+2., iteration*3.+2.)
        assert s.stage == 'SEARCH'
    row = log.finish(9., 9., outcome='failed', reason='offline_test_limit', robot_xy=xy, source_end=None)
    assert row['method_id'] == method and row['schema_version'] == 2
    assert row['attempt_count'] == row['measurement_count'] == 3
    assert row['search_iterations'] == s.search_iterations == 2
    assert row['initial_seconds'] + row['search_seconds'] == row['total_seconds']
    assert row['final_position_error'] is None
    saved = next(csv.DictReader((log.directory/'runs.csv').open()))
    assert saved['measurement_count'] == '3'
    assert len(set(path)) == 3
    fresh = HrsSession(method, ms)
    assert fresh.stage == 'INITIAL' and fresh.attempt_count == 0
    assert not fresh.spiral.visited and fresh.spiral.center is None
    assert initial_visited == {g.variable_at(1, 1), g.variable_at(3, 3)}


@pytest.mark.parametrize('method', list(METHODS))
def test_ros_planning_dispatch_uses_method_and_initial_revisit(method, map_message_factory):
    from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
    g, o = setup_grid(map_message_factory)
    sampled = {g.variable_at(1, 1)}
    c = NS(hrs_session=HrsSession(method, [measurement(.75, .75, .8)]),
           sampled_variables=sampled, gmrf=g, hrs_ucb_k=.2,
           hrs_distance_weight=.03, hrs_manager=HrsManager(),
           navigation_goal_variables=set(range(len(g.solution))),
           sampling_distance=o, latest_pose=NS(pose=NS(position=NS(x=.25, y=.25))),
           planning_executor=NS(active_task=None), hrs_candidate_variables=set(),
           hrs_cycles=0, publish_phase=lambda phase: None,
           publish_candidates=lambda *args: None, publish_hrs_route=lambda *args: None,
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=1)),
           get_logger=lambda: NS(info=lambda msg: None))
    workflow = HrsWorkflow(c)
    c.build_candidates = workflow.build_candidates
    sent = []
    c.send_current_goal = lambda: sent.append(c.active_hrs_target)
    workflow.start_hrs_planning()
    assert len(sent) == 1
    if method in ('M1', 'M2', 'M5', 'M6'):
        assert sent[0].variable in sampled
    else:
        assert sent[0].variable not in sampled


@pytest.mark.parametrize('update_ok,value', [(True, 1.), (True, .4), (False, 1.)])
def test_initial_stage_waits_for_gmrf_before_termination(update_ok, value, map_message_factory, tmp_path):
    from icir_cleanroom.gas_mapping.ros.navigation_workflow import NavigationWorkflow
    g, o = setup_grid(map_message_factory)
    session = HrsSession('M3')
    candidates = HrsManager().build_candidates(g, set(), .2)
    _, target = session.select(candidates, gmrf=g, current_xy=(.25, .25), oracle=o, distance_weight=.03)
    log = HrsRunLog(tmp_path, 'offline_test_clock')
    log.start(0., 0., event_id='test', lrs_lap=1, source=None, parameters={})
    result = NS(phase='HRS_NAVIGATION', mean=value, sample_count=10,
                pose=NS(pose=NS(position=NS(x=target.x, y=target.y))))
    calls = []
    c = NS(phase='HRS_NAVIGATION', hrs_session=session, hrs_run_log=log,
           hrs_log_source=None, gmrf=g, latest_pose=result.pose,
           active_hrs_target=target, dwell_timer=None, hrs_manager=HrsManager(),
           hrs_gmrf_dirty=False,
           measurement_manager=NS(finish=lambda gen: result, commit=lambda *args: NS(timestamp=1.)),
           history=NS(update=lambda *args: None), publish_history=lambda: None,
           record_event_measurement=lambda *args: None,
           publish_measurements=lambda: None,
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=2_000_000_000)),
           get_logger=lambda: NS(info=lambda msg: None, warning=lambda msg: None, error=lambda msg: None))
    def update(*args, **kwargs):
        assert session.stage == 'INITIAL' and log.active['first_ros'] is None
        calls.append('update')
        return update_ok
    c.update_gmrf = update
    c.finalize_hrs_gmrf_batch = lambda *args: update_ok
    c.finish_hrs_search = lambda reason: calls.append(('failed', reason))
    c.advance_after_target = lambda: calls.append('next')
    NavigationWorkflow(c).finish_dwell(1)
    assert calls[0] == 'update'
    if not update_ok:
        assert session.stage == 'INITIAL' and log.active['first_ros'] is None
        assert calls[-1] == ('failed', 'GMRF update failed')
    else:
        # No fixed concentration threshold gates termination any more:
        # once the GMRF update succeeds, the workflow always moves on
        # regardless of the measured value.
        assert session.stage == 'SEARCH' and log.active['first_ros'] is not None
        assert calls[-1] == 'next'
