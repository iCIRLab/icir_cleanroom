import csv
from types import SimpleNamespace as NS
import pytest
from builtin_interfaces.msg import Time
from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.ros.hrs_run_logging import record_hrs, source_position


def begin(tmp_path, source=(2., 3.)):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start(100., 1000., event_id='event-1', lrs_lap=1,
              source=source, parameters={'hrs_search_patience': 10})
    return log


def finish(log, reason='test_reason', source=(2., 3.)):
    return log.finish(120., 1030., reason=reason,
                      robot_xy=(2.1, 3.), source_end=source)


@pytest.mark.parametrize('reason', ['no candidates', 'maximum attempts', 'node_shutdown'])
def test_distance_sums_travel_including_return_path_and_resets(tmp_path, reason):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.position((100., 100.))  # LRS travel is excluded.
    log.start(100., 1000., event_id='event-1', lrs_lap=1,
              source=None, parameters={}, robot_xy=(0., 0.))
    log.position((3., 4.))
    log.position((3., 4.))  # Stationary samples add no distance.
    log.position((0., 0.))  # Returning to the start still adds travel.
    row = log.finish(120., 1030., reason=reason,
                     robot_xy=(0., 2.), source_end=None)
    assert row['hrs_distance_m'] == pytest.approx(12.)
    assert row['total_distance_m'] is None  # No preceding LRS record.
    log.position((100., 100.))  # Travel after HRS completion is excluded.
    saved = next(csv.DictReader((log.directory/'runs.csv').open()))
    assert float(saved['hrs_distance_m']) == pytest.approx(12.)
    log.start(200., 1100., event_id='event-2', lrs_lap=2,
              source=None, parameters={}, robot_xy=(50., 50.))
    row = log.finish(210., 1110., reason='stationary',
                     robot_xy=(50., 50.), source_end=None)
    assert row['hrs_distance_m'] == 0.


def test_distance_handles_missing_and_nonfinite_positions(tmp_path):
    log = begin(tmp_path)
    log.position((float('nan'), 0.))
    row = log.finish(120., 1030., reason='no pose',
                     robot_xy=None, source_end=None)
    assert row['hrs_distance_m'] is None
    saved = next(csv.DictReader((log.directory/'runs.csv').open()))
    assert saved['hrs_distance_m'] == ''
    log.start(200., 1100., event_id='event-2', lrs_lap=2,
              source=None, parameters={})
    log.position(None)
    log.position((1., 1.))
    log.position((float('inf'), 2.))
    log.position((4., 5.))
    row = log.finish(210., 1110., reason='test',
                     robot_xy=(4., 5.), source_end=None)
    assert row['hrs_distance_m'] == pytest.approx(5.)


def test_ros_pose_callbacks_accumulate_only_between_start_and_finish(tmp_path):
    from icir_cleanroom.gas_mapping.config import ControllerConfig
    from icir_cleanroom.gas_mapping.ros.controller_node import GasMappingControllerNode
    log = HrsRunLog(tmp_path, 'ros_simulation')
    c = NS(hrs_run_log=log, hrs_log_source=None, current_event_id='event-1',
           lrs_lap=1, config=ControllerConfig.defaults(),
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=100_000_000_000)),
           get_logger=lambda: NS(info=lambda msg: None, error=lambda msg: None))
    def pose(x, y):
        GasMappingControllerNode.pose_callback(c, NS(pose=NS(position=NS(x=x, y=y))))
    pose(99., 99.)
    pose(0., 0.)
    record_hrs(c, 'start')
    pose(3., 4.)
    pose(6., 4.)
    row = record_hrs(c, 'finish', reason='test')
    assert row['hrs_distance_m'] == pytest.approx(8.)
    pose(99., 99.)
    assert log.active is None
    assert float(next(csv.DictReader((log.directory/'runs.csv').open()))[
        'hrs_distance_m']) == pytest.approx(8.)


def test_time_split_includes_selection_navigation_and_first_dwell(tmp_path):
    log = begin(tmp_path)
    log.event('target_selected', 102., 1003., x=2., y=3.)
    log.measurement(110., 1012., xy=(2.1, 3.), value=.91, sample_count=20)
    log.initial_complete(112., 1015.)
    row = finish(log)
    assert (row['initial_seconds'], row['search_seconds'], row['total_seconds']) == (12., 8., 20.)
    assert (row['wall_initial_seconds'], row['wall_search_seconds'], row['wall_total_seconds']) == (15., 15., 30.)
    assert row['source_position_error'] == pytest.approx(.1)
    assert 'outcome' not in row and 'gas_source_detected' not in row
    assert 'gas_source_measurement_failed' not in row
    assert row['estimated_source_x'] == 2.1
    rows = list(csv.DictReader((log.directory/'runs.csv').open()))
    assert len(rows) == 1
    assert float(rows[0]['total_seconds']) == float(rows[0]['initial_seconds'])+float(rows[0]['search_seconds'])
    assert rows[0]['termination_reason'] == row['termination_reason']
    events = list(csv.DictReader((log.directory/'events.csv').open()))
    assert [e['event'] for e in events] == ['lrs_complete_hrs_start', 'target_selected', 'measurement_complete', 'initial_gmrf_complete', 'hrs_end']
    assert finish(log) is None  # source transition/shutdown cannot duplicate row


def test_first_measurement_has_zero_search_if_end_is_immediate(tmp_path):
    log=begin(tmp_path)
    log.measurement(120., 1030., xy=(2.,3.), value=1., sample_count=20)
    log.initial_complete(120., 1030.)
    row=finish(log)
    assert row['initial_seconds']==20. and row['search_seconds']==0.


def test_end_before_first_measurement_does_not_invent_search_time(tmp_path):
    log=begin(tmp_path)
    row=finish(log,'no candidates')
    assert row['initial_seconds']==20.
    assert row['search_seconds'] is None
    assert not row['initial_measurement_completed']
    assert row['measurement_count']==0
    assert row['estimated_source_x'] is None
    assert row['final_measured_concentration'] is None
    saved=next(csv.DictReader((log.directory/'runs.csv').open()))
    assert saved['search_seconds']=='' and saved['final_measured_concentration']==''


def test_estimate_and_error_are_recorded_without_source_classification(tmp_path):
    log=begin(tmp_path)
    log.measurement(110., 1010., xy=(4.,5.), value=.2, sample_count=10)
    row=finish(log,'no candidates')
    assert row['last_measurement_x']==4.
    assert row['final_robot_x']==2.1
    assert row['estimated_source_x'] == 4.
    assert row['source_position_error'] == pytest.approx(8. ** .5)


def test_estimate_matches_final_cell_and_retains_raw_measurement(tmp_path):
    snap_calls = []
    def to_cell_center(x, y):
        snap_calls.append((x, y))
        return (round(x), round(y))
    log = begin(tmp_path)
    log.measurement(110., 1010., xy=(2.1, 3.4), value=.91, sample_count=20)
    row = log.finish(120., 1030., reason='ok',
                      robot_xy=(2.1, 3.4), source_end=(2., 3.),
                      to_cell_center=to_cell_center)
    assert row['estimated_source_x'] == 2 and row['estimated_source_y'] == 3
    assert row['best_measurement_x'] == 2.1 and row['best_measurement_y'] == 3.4
    assert row['source_position_error'] == 0.
    assert (row['estimated_source_cell_x'], row['estimated_source_cell_y']) == (2, 3)
    assert snap_calls == [(2.1, 3.4)]  # snapped the same point logged raw above


def test_estimated_source_cell_comes_from_best_measurement(tmp_path):
    log = begin(tmp_path)
    log.measurement(110., 1010., xy=(4.6, 5.4), value=.2, sample_count=10)
    row = log.finish(120., 1030., reason='no candidates',
                      robot_xy=(4.6, 5.4), source_end=None,
                      to_cell_center=lambda x, y: (round(x), round(y)))
    assert row['estimated_source_x'] == 5
    assert (row['estimated_source_cell_x'], row['estimated_source_cell_y']) == (5, 5)


def test_estimated_source_cell_uses_the_highest_measured_point_not_the_last(tmp_path):
    log = begin(tmp_path)
    log.measurement(105., 1005., xy=(-4.0, -7.0), value=.976, sample_count=38)
    log.measurement(110., 1010., xy=(4.6, 5.4), value=.2, sample_count=10)
    row = log.finish(120., 1030., reason='no relative improvement',
                      robot_xy=(4.6, 5.4), source_end=None,
                      to_cell_center=lambda x, y: (round(x), round(y)))
    assert (row['last_measurement_x'], row['last_measurement_y']) == (4.6, 5.4)
    assert row['final_measured_concentration'] == .2
    assert (row['best_measurement_x'], row['best_measurement_y']) == (-4.0, -7.0)
    assert row['best_measured_concentration'] == .976
    assert (row['estimated_source_cell_x'], row['estimated_source_cell_y']) == (-4, -7)


def test_estimated_source_cell_is_blank_without_a_resolver(tmp_path):
    log = begin(tmp_path)
    log.measurement(110., 1010., xy=(2.1, 3.4), value=.91, sample_count=20)
    row = finish(log)  # to_cell_center omitted; default keeps this class grid-agnostic
    assert row['estimated_source_cell_x'] is None and row['estimated_source_cell_y'] is None


def test_missing_source_position_remains_unevaluated(tmp_path):
    log=begin(tmp_path,None)
    log.measurement(110., 1010., xy=(4.,5.), value=1., sample_count=10)
    row=finish(log,source=None)
    assert row['estimated_source_x'] == 4.
    assert not row['source_position_available']
    assert row['source_position_error'] is None


def test_repeated_trials_reset_clocks_and_measurements(tmp_path):
    log=begin(tmp_path)
    log.measurement(110., 1010., xy=(4.,5.), value=1., sample_count=10)
    first=finish(log)
    log.start(200., 1100., event_id='event-2', lrs_lap=2, source=(8.,9.),parameters={})
    second=log.finish(203., 1104., reason='no candidates',robot_xy=(4.,5.),source_end=(8.,9.))
    assert first['run_id']!=second['run_id']
    assert second['measurement_count']==0 and second['total_seconds']==3.
    assert second['source_start_x']==8.
    assert len(list(csv.DictReader((log.directory/'runs.csv').open())))==2


def test_clock_reset_does_not_publish_misleading_ros_durations(tmp_path):
    log=begin(tmp_path)
    log.event('target_selected', 1., 1002.)
    row=finish(log)
    assert not row['ros_clock_valid'] and row['total_seconds'] is None
    assert row['wall_total_seconds']==30.


def test_shutdown_records_reason_without_source_classification(tmp_path):
    log=begin(tmp_path)
    row=finish(log,'node_shutdown')
    assert row['termination_reason']=='node_shutdown'
    assert 'outcome' not in row
    assert 'gas_source_measurement_failed' not in row


def test_truth_only_changes_posthoc_error_not_times_or_estimate(tmp_path):
    rows=[]
    for source in [(2.,3.),(100.,100.)]:
        log=begin(tmp_path,source)
        log.measurement(110., 1010., xy=(2.,3.), value=.95, sample_count=20)
        rows.append(finish(log,source=source))
    for key in ['termination_reason','total_seconds','estimated_source_x']:
        assert rows[0][key]==rows[1][key]
    assert rows[0]['source_position_error']!=rows[1]['source_position_error']


def test_source_marker_delete_and_wrong_frame_are_not_real_sources():
    c=NS(hrs_log_source=None)
    marker=NS(action=0, header=NS(frame_id='map'),pose=NS(position=NS(x=2.,y=3.)))
    source_position(c,marker);assert c.hrs_log_source==(2.,3.)
    marker.action=2;source_position(c,marker);assert c.hrs_log_source is None
    marker.action=0;marker.header.frame_id='odom'
    source_position(c,marker);assert c.hrs_log_source is None


def test_finish_hrs_search_saves_row_before_advancing_source(tmp_path):
    from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
    log=begin(tmp_path)
    log.measurement(110., 1010., xy=(2.1,3.), value=.95, sample_count=20)
    calls=[]
    c=NS(hrs_run_log=log,hrs_log_source=(2.,3.),latest_pose=NS(pose=NS(position=NS(x=2.1,y=3.))),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=120_000_000_000,to_msg=lambda:Time())),
         get_logger=lambda:NS(info=lambda msg:None,warning=lambda msg:None,error=lambda msg:None),
         gmrf=NS(geometry=NS(
             world_to_cell=lambda x,y:(0,0),cell_center=lambda r,c:(2.0,3.0))),
         repeat_after_hrs=True,
         navigation_manager=NS(issue_goal=lambda:1),
         nav2=NS(send=lambda pose,gen,on_success,on_failure:on_success()),
         publish_estimated_source=lambda xy:None)
    def start_source_transition(reason):
        assert log.active is None
        row=next(csv.DictReader((log.directory/'runs.csv').open()))
        assert row['source_x']=='2.0' and 'outcome' not in row
        assert row['estimated_source_cell_x']=='2.0' and row['estimated_source_cell_y']=='3.0'
        c.hrs_log_source=(99.,99.)
        calls.append('transition')
    c.start_source_transition=start_source_transition
    HrsWorkflow(c).finish_hrs_search('no relative improvement in 10 consecutive HRS attempts')
    assert calls==['transition']


def test_finish_hrs_search_completes_even_if_the_final_approach_fails(tmp_path):
    from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
    log=begin(tmp_path)
    log.measurement(110., 1010., xy=(2.1,3.), value=.95, sample_count=20)
    calls=[]
    c=NS(hrs_run_log=log,hrs_log_source=(2.,3.),latest_pose=NS(pose=NS(position=NS(x=2.1,y=3.))),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=120_000_000_000,to_msg=lambda:Time())),
         get_logger=lambda:NS(info=lambda msg:None,warning=lambda msg:calls.append(('warn',msg)),error=lambda msg:None),
         gmrf=NS(geometry=NS(
             world_to_cell=lambda x,y:(0,0),cell_center=lambda r,c:(2.0,3.0))),
         repeat_after_hrs=False,
         navigation_manager=NS(issue_goal=lambda:1),
         nav2=NS(send=lambda pose,gen,on_success,on_failure:on_failure('goal rejected')),
         publish_estimated_source=lambda xy:None,
         complete_mapping=lambda reason:calls.append(('complete',reason)))
    HrsWorkflow(c).finish_hrs_search('maximum 100 HRS target attempts reached')
    assert calls[-1]==('complete','HRS terminated: maximum 100 HRS target attempts reached; final approach stopped: goal rejected')


def test_lrs_boundary_is_recorded_before_hrs_selection(tmp_path):
    from icir_cleanroom.gas_mapping.ros.lrs_workflow import LrsWorkflow
    from icir_cleanroom.gas_mapping.config import ControllerConfig
    log=HrsRunLog(tmp_path,'ros_simulation')
    c=NS(hrs_run_log=log,hrs_log_source=None,current_event_id='event-1',
         config=ControllerConfig.defaults(),lrs_lap=1,lap_hazard_detected=True,
         lap_max_concentration=.5,hazard_threshold=.2,
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=100_000_000_000)),
         get_logger=lambda:NS(info=lambda msg:None,error=lambda msg:None),
         persist_history=lambda msg:None, publish_phase=lambda phase:None)
    calls=[]
    def planning():
        assert log.active['start_ros']==100.
        calls.append('plan')
    c.start_hrs_planning=planning
    LrsWorkflow(c).finish_lrs_navigation()
    assert calls==['plan']


@pytest.mark.parametrize('ending', ['arrived', 'navigation_stopped', 'shutdown'])
def test_final_approach_is_included_and_runner_cannot_finish_early(tmp_path, ending):
    from icir_cleanroom.gas_mapping.ros.controller_node import GasMappingControllerNode
    from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
    import json

    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start(100., 1000., event_id='event-1', lrs_lap=1,
              source=(3., 5.), parameters={}, robot_xy=(0., 0.))
    log.measurement(105., 1005., xy=(3.1, 4.1), value=.9, sample_count=10)
    log.initial_complete(106., 1006.)
    log.measurement(110., 1010., xy=(0., 4.), value=.2, sample_count=10)
    log.position((0., 4.))
    now = [120.]
    sent, completed, markers = [], [], []
    c = NS(hrs_run_log=log, hrs_log_source=(3., 5.),
           latest_pose=NS(pose=NS(position=NS(x=0., y=4.))),
           get_clock=lambda: NS(now=lambda: NS(
               nanoseconds=int(now[0]*1e9), to_msg=lambda: Time())),
           get_logger=lambda: NS(info=lambda msg: None, warning=lambda msg: None,
                                 error=lambda msg: pytest.fail(msg)),
           gmrf=NS(geometry=NS(world_to_cell=lambda x,y:(0,0),
                              cell_center=lambda r,c:(3.,4.))),
           repeat_after_hrs=False, navigation_manager=NS(issue_goal=lambda: 1),
           nav2=NS(send=lambda pose, gen, arrived, stopped: sent.append((pose, arrived, stopped))),
           publish_estimated_source=markers.append,
           complete_mapping=completed.append)
    workflow = HrsWorkflow(c)
    workflow.finish_hrs_search('no relative improvement')
    workflow.finish_hrs_search('duplicate call')
    assert len(sent) == 1 and completed == []
    assert markers == [(3., 4.)]
    assert log.active is not None
    assert not (log.directory/'runs.csv').exists()
    assert (sent[0][0].pose.position.x, sent[0][0].pose.position.y) == (3., 4.)

    # Travel after estimate selection must be included; the estimate stays fixed.
    now[0] = 125.
    GasMappingControllerNode.pose_callback(c, NS(pose=NS(position=NS(x=2., y=4.))))
    now[0] = 130.
    if ending == 'arrived':
        GasMappingControllerNode.pose_callback(c, NS(pose=NS(position=NS(x=3., y=4.))))
        sent[0][1]()
        sent[0][1]()  # A duplicate result must not write or complete twice.
    elif ending == 'navigation_stopped':
        sent[0][2]('goal rejected')
    else:
        record_hrs(c, 'finish', reason='node_shutdown')

    rows = list(csv.DictReader((log.directory/'runs.csv').open()))
    assert len(rows) == 1 and log.active is None
    row = rows[0]
    assert float(row['total_seconds']) == 30.
    assert float(row['initial_seconds']) == 6.
    assert float(row['search_seconds']) == 24.
    assert float(row['hrs_distance_m']) == (7. if ending == 'arrived' else 6.)
    assert (row['estimated_source_x'], row['estimated_source_y']) == ('3.0', '4.0')
    assert float(row['source_position_error']) == 1.
    assert len(completed) == (0 if ending == 'shutdown' else 1)
    if ending == 'navigation_stopped':
        assert 'final approach stopped: goal rejected' in row['termination_reason']
    if ending == 'shutdown':
        assert row['termination_reason'] == 'node_shutdown'
    assert {'outcome', 'gas_source_detected', 'gas_source_measurement_failed'}.isdisjoint(row)
    events = list(csv.DictReader((log.directory/'events.csv').open()))
    assert events[-1]['event'] == 'hrs_end'
    assert float(events[-1]['ros_seconds']) == 130.
    assert 'outcome' not in json.loads(events[-1]['details'])
    history = list(csv.DictReader((log.directory/'estimate_history.csv').open()))
    assert [r['event'] for r in history[-2:]] == ['final_estimate_selected', 'hrs_end']
    assert float(history[-1]['elapsed_seconds']) == float(row['total_seconds'])
    assert float(history[-1]['hrs_distance_m']) == float(row['hrs_distance_m'])
    assert float(history[-1]['source_position_error']) == float(row['source_position_error'])


def test_final_estimate_is_frozen_before_travel_and_reset_for_next_run(tmp_path):
    log = begin(tmp_path)
    log.measurement(110., 1010., xy=(2.1, 3.4), value=.9, sample_count=10)
    first = log.final_estimate(lambda x,y:(2., 3.))
    first['estimated_source_x'] = 99.  # Returned data cannot mutate the estimate.
    row = log.finish(120., 1030., reason='done', robot_xy=(2.2,3.),
                     source_end=(2.,3.), to_cell_center=lambda x,y:(99.,99.))
    assert row['estimated_source_x'] == 2. and row['source_position_error'] == 0.
    log.start(200., 1100., event_id='event-2', lrs_lap=2, source=(2.,3.), parameters={})
    row = log.finish(201., 1101., reason='no candidates', robot_xy=(2.,3.), source_end=(2.,3.))
    assert row['estimated_source_x'] is None and row['source_position_error'] is None


def read_events(log):
    import csv
    with (log.directory/'events.csv').open(encoding='utf-8', newline='') as s:
        return list(csv.DictReader(s))


def events_of(log, kind):
    import json
    return [json.loads(r['details'] or '{}')
            for r in read_events(log) if r['event'] == kind]


@pytest.mark.parametrize('succeeded', [True, False])
def test_final_approach_records_its_goal_outcome_and_arrival(tmp_path, succeeded):
    """Arrival time, arrival position and success must not need parsing a reason."""
    from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
    log = begin(tmp_path)
    log.measurement(110., 1010., xy=(2.1, 3.), value=.95, sample_count=20)
    c = NS(hrs_run_log=log, hrs_log_source=(2., 3.),
           latest_pose=NS(pose=NS(position=NS(x=2.04, y=2.97))),
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=120_000_000_000,
                                               to_msg=lambda: Time())),
           get_logger=lambda: NS(info=lambda msg: None, warning=lambda msg: None,
                                 error=lambda msg: pytest.fail(msg)),
           gmrf=NS(geometry=NS(world_to_cell=lambda x, y: (0, 0),
                               cell_center=lambda r, c: (2.0, 3.0))),
           repeat_after_hrs=False, navigation_manager=NS(issue_goal=lambda: 1),
           nav2=NS(send=lambda pose, gen, on_success, on_failure:
                   on_success() if succeeded else on_failure('status=6')),
           publish_estimated_source=lambda xy: None,
           complete_mapping=lambda reason: None)
    HrsWorkflow(c).finish_hrs_search('no relative improvement in 10 consecutive HRS attempts')

    started = events_of(log, 'final_approach_start')
    ended = events_of(log, 'final_approach_end')
    assert len(started) == 1 and len(ended) == 1
    assert (started[0]['goal_x'], started[0]['goal_y']) == (2.0, 3.0)
    assert ended[0]['succeeded'] is succeeded
    assert ended[0]['reason'] == ('arrived' if succeeded else 'status=6')
    assert (ended[0]['robot_x'], ended[0]['robot_y']) == (2.04, 2.97)
    # The arrival is timestamped like any other event.
    stamps = [float(r['ros_seconds']) for r in read_events(log)
              if r['event'] == 'final_approach_end']
    assert stamps == pytest.approx([120.])


def test_navigation_events_name_the_cell_they_refer_to():
    """A navigation failure is only actionable with the goal cell attached."""
    from icir_cleanroom.gas_mapping.ros.navigation_workflow import target_details
    target = NS(variable=7, x=1.5, y=-2.5, row=3, col=4)
    assert target_details(NS(active_hrs_target=target)) == dict(
        target_variable=7, target_x=1.5, target_y=-2.5,
        target_row=3, target_col=4)
    # No target set (for example a final approach) leaves the fields blank
    # rather than dropping them, so the CSV column stays stable.
    assert target_details(NS(active_hrs_target=None)) == dict(
        target_variable=None, target_x=None, target_y=None,
        target_row=None, target_col=None)
