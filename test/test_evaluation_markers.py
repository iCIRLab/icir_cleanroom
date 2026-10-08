"""RViz uses the exact logged reference and keeps the last complete snapshot."""
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray

from icir_cleanroom.gas_mapping.application.benchmark import EvaluationReference
from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.ros.evaluation_markers import evaluation_markers
from icir_cleanroom.gas_mapping.ros.hrs_run_logging import record_hrs, record_lrs, source_position
from icir_cleanroom.gas_mapping.ros.publishers import ControllerVisualization, PUBLISHER_SPECS


def reference_row(estimate=None):
    source = (-2., 6.)
    ref = EvaluationReference(source, ((-3., 6.5), (-1., 6.5)), True,
                              'source_outside_goal_domain', math.sqrt(1.25))
    return dict(ref.evaluate(estimate, source),
                estimated_source_x=estimate[0] if estimate else None,
                estimated_source_y=estimate[1] if estimate else None)


def by_namespace(message, namespace):
    return [m for m in message.markers if m.ns == namespace]


def test_both_seed71_candidates_visible_before_an_estimate_exists():
    msg = evaluation_markers(reference_row(), Time(sec=1))
    assert msg.markers[0].action == Marker.DELETEALL
    cells = by_namespace(msg, 'evaluation_candidates')[0]
    assert [(p.x, p.y) for p in cells.points] == [(-3., 6.5), (-1., 6.5)]
    assert cells.type == Marker.CUBE_LIST
    assert all(m.type != Marker.TEXT_VIEW_FACING for m in msg.markers)
    assert not by_namespace(msg, 'evaluation_error')
    assert all(m.header.frame_id == 'map' and m.lifetime.sec == 0
               and m.lifetime.nanosec == 0 for m in msg.markers)


@pytest.mark.parametrize('estimate,nearest', [((-1., 7.), (-1., 6.5)),
                                             ((-3., 6.5), (-3., 6.5)),
                                             ((-2., 6.5), (-3., 6.5))])
def test_error_line_uses_nearest_candidate_and_logged_error(estimate, nearest):
    row = reference_row(estimate)
    msg = evaluation_markers(row, Time())
    line = by_namespace(msg, 'evaluation_error')[0]
    assert [(p.x, p.y) for p in line.points] == [estimate, nearest]
    assert math.dist(estimate, nearest) == row['adjusted_source_position_error']
    assert all(m.type != Marker.TEXT_VIEW_FACING for m in msg.markers)
    assert len(by_namespace(msg, 'evaluation_candidates')[0].points) == 2


@pytest.mark.parametrize('reason', ['missing_source', 'source_changed', 'missing_map', 'no_reachable_cells'])
def test_unevaluated_reference_clears_old_markers(reason):
    row = reference_row((-1., 7.))
    row.update(evaluation_reason=reason, evaluation_available=False)
    msg = evaluation_markers(row, Time())
    assert len(msg.markers) == 1 and msg.markers[0].action == Marker.DELETEALL


def test_no_correction_or_new_run_clears_prior_reference():
    row = reference_row((-1., 7.))
    row['evaluation_reference_adjusted'] = False
    for data in (None, row):
        assert [m.action for m in evaluation_markers(data, Time()).markers] == [Marker.DELETEALL]


def test_logging_hooks_match_csv_and_retain_final_display(tmp_path):
    messages = []
    tick = [0]
    grid = GridGeometry(3, 3, .5, -.25, -.25)
    mask = np.ones((3, 3), bool)
    mask[1, 1] = False
    clock = NS(now=lambda: NS(nanoseconds=tick[0]*10**9, to_msg=lambda: Time(sec=tick[0])))
    c = NS(hrs_run_log=HrsRunLog(tmp_path, 'ros_simulation'), hrs_log_source=(.5, .5),
           current_event_id='event', lrs_lap=1, method='M6', latest_pose=None,
           lap_hazard_detected=True, get_clock=lambda: clock,
           get_logger=lambda: NS(info=lambda *a: None, error=lambda e: pytest.fail(e)),
           config=NS(flat_values=lambda: {'method':'M6'}),
           navigation_goal_geometry=grid, navigation_goal_mask=mask,
           gmrf=NS(geometry=grid, resolution=.5),
           hrs_evaluation_reference_pub=NS(publish=messages.append))
    c.publish_evaluation_reference = ControllerVisualization(c).publish_evaluation_reference
    record_lrs(c, 'start')
    assert messages[-1].markers[0].action == Marker.DELETEALL
    tick[0] = 10
    record_lrs(c, 'finish', outcome='completed', reason='lap')
    record_hrs(c, 'start')
    assert len(by_namespace(messages[-1], 'evaluation_candidates')[0].points) == 4
    mask[:] = False  # A live change must not change the frozen displayed candidates.
    tick[0] = 20
    record_hrs(c, 'measurement', xy=(.04, .52), value=1., sample_count=1)
    line = by_namespace(messages[-1], 'evaluation_error')[0]
    assert (line.points[0].x, line.points[0].y) == (0., .5)  # Same cell snap as CSV.
    record_hrs(c, 'estimate')
    tick[0] = 30
    row = record_hrs(c, 'finish', reason='done')
    history = list(csv.DictReader((c.hrs_run_log.directory/'estimate_history.csv').open()))
    assert float(history[-1]['adjusted_source_position_error']) == row['adjusted_source_position_error'] == 0
    cells = by_namespace(messages[-1], 'evaluation_candidates')[0]
    assert [[p.x, p.y] for p in cells.points] == json.loads(row['evaluation_reference_candidates'])
    last = messages[-1]
    record_hrs(c, 'finish', reason='node_shutdown')
    assert messages[-1] is last  # The ordinary shutdown hook cannot erase final markers.
    source = Marker()
    source.header.frame_id = 'map'
    source.pose.position.x = source.pose.position.y = .5
    source_position(c, source)
    assert messages[-1] is last  # Repeated truth publication also keeps the result.
    source.action = Marker.DELETE
    source_position(c, source)
    assert [m.action for m in messages[-1].markers] == [Marker.DELETEALL]


def test_rviz_enables_retained_reference_topic():
    topic = '/gas_mapping/hrs/evaluation_reference'
    assert ('hrs_evaluation_reference_pub', MarkerArray, topic) in PUBLISHER_SPECS
    path = Path(__file__).resolve().parents[1]/'rviz/cleanroom_empty.rviz'
    displays = yaml.safe_load(path.read_text())['Visualization Manager']['Displays']
    display = next(d for d in displays if d.get('Name') == 'HRSEvaluationReference')
    assert display['Enabled'] and display['Class'] == 'rviz_default_plugins/MarkerArray'
    assert display['Topic']['Value'] == topic
    assert display['Topic']['Durability Policy'] == 'Transient Local'
