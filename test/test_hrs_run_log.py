import csv
from types import SimpleNamespace as NS
import pytest
from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.ros.hrs_run_logging import record_hrs, source_position


def begin(tmp_path, source=(2., 3.)):
    log = HrsRunLog(tmp_path, 'ros_simulation')
    log.start(100., 1000., event_id='event-1', lrs_lap=1,
              source=source, parameters={'hrs_response_threshold': .9})
    return log


def finish(log, outcome='detected', source=(2., 3.)):
    return log.finish(120., 1030., outcome=outcome, reason='test_reason',
                      robot_xy=(2.1, 3.), source_end=source)


def test_time_split_includes_selection_navigation_and_first_dwell(tmp_path):
    log = begin(tmp_path)
    log.event('target_selected', 102., 1003., x=2., y=3.)
    log.measurement(110., 1012., xy=(2.1, 3.), value=.91, sample_count=20)
    row = finish(log)
    assert (row['initial_seconds'], row['search_seconds'], row['total_seconds']) == (10., 10., 20.)
    assert (row['wall_initial_seconds'], row['wall_search_seconds'], row['wall_total_seconds']) == (12., 18., 30.)
    assert row['source_position_error'] == pytest.approx(.1)
    assert row['gas_source_detected'] and not row['gas_source_measurement_failed']
    assert row['estimated_source_x'] == 2.1
    rows = list(csv.DictReader((log.directory/'runs.csv').open()))
    assert len(rows) == 1
    assert float(rows[0]['total_seconds']) == float(rows[0]['initial_seconds'])+float(rows[0]['search_seconds'])
    assert rows[0]['outcome'] == row['outcome']
    events = list(csv.DictReader((log.directory/'events.csv').open()))
    assert [e['event'] for e in events] == ['lrs_complete_hrs_start', 'target_selected', 'measurement_complete', 'hrs_end']
    assert finish(log) is None  # source transition/shutdown cannot duplicate row


def test_first_measurement_success_has_zero_search_if_decision_is_immediate(tmp_path):
    log=begin(tmp_path)
    log.measurement(120., 1030., xy=(2.,3.), value=1., sample_count=20)
    row=finish(log)
    assert row['initial_seconds']==20. and row['search_seconds']==0.


def test_failure_before_first_measurement_does_not_invent_search_time(tmp_path):
    log=begin(tmp_path)
    row=finish(log,'failed')
    assert row['initial_seconds']==20.
    assert row['search_seconds'] is None
    assert not row['initial_measurement_completed']
    assert row['measurement_count']==0
    assert row['gas_source_measurement_failed']
    assert row['estimated_source_x'] is None
    assert row['final_measured_concentration'] is None
    saved=next(csv.DictReader((log.directory/'runs.csv').open()))
    assert saved['search_seconds']=='' and saved['final_measured_concentration']==''


def test_failed_search_logs_last_measurement_without_claiming_it_is_source(tmp_path):
    log=begin(tmp_path)
    log.measurement(110., 1010., xy=(4.,5.), value=.2, sample_count=10)
    row=finish(log,'failed')
    assert row['last_measurement_x']==4.
    assert row['final_robot_x']==2.1
    assert row['estimated_source_x'] is None
    assert row['source_position_error'] is None


def test_missing_source_position_remains_unevaluated(tmp_path):
    log=begin(tmp_path,None)
    log.measurement(110., 1010., xy=(4.,5.), value=1., sample_count=10)
    row=finish(log,source=None)
    assert row['gas_source_detected']
    assert not row['source_position_available']
    assert row['source_position_error'] is None


def test_repeated_trials_reset_clocks_and_measurements(tmp_path):
    log=begin(tmp_path)
    log.measurement(110., 1010., xy=(4.,5.), value=1., sample_count=10)
    first=finish(log)
    log.start(200., 1100., event_id='event-2', lrs_lap=2, source=(8.,9.),parameters={})
    second=log.finish(203., 1104., outcome='failed', reason='no candidates',robot_xy=(4.,5.),source_end=(8.,9.))
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


def test_interruption_is_separate_from_algorithm_failure(tmp_path):
    log=begin(tmp_path)
    row=finish(log,'interrupted')
    assert row['outcome']=='interrupted'
    assert row['gas_source_measurement_failed'] is None


def test_truth_only_changes_posthoc_error_not_times_or_outcome(tmp_path):
    rows=[]
    for source in [(2.,3.),(100.,100.)]:
        log=begin(tmp_path,source)
        log.measurement(110., 1010., xy=(2.,3.), value=.95, sample_count=20)
        rows.append(finish(log,source=source))
    for key in ['outcome','total_seconds','estimated_source_x','gas_source_measurement_failed']:
        assert rows[0][key]==rows[1][key]
    assert rows[0]['source_position_error']!=rows[1]['source_position_error']


def test_source_marker_delete_and_wrong_frame_are_not_real_sources():
    c=NS(hrs_log_source=None)
    marker=NS(action=0, header=NS(frame_id='map'),pose=NS(position=NS(x=2.,y=3.)))
    source_position(c,marker);assert c.hrs_log_source==(2.,3.)
    marker.action=2;source_position(c,marker);assert c.hrs_log_source is None
    marker.action=0;marker.header.frame_id='odom'
    source_position(c,marker);assert c.hrs_log_source is None


def test_success_is_saved_before_source_transition(tmp_path):
    from icir_cleanroom.gas_mapping.ros.hrs_workflow import HrsWorkflow
    from icir_cleanroom.gas_mapping.application.hrs import HrsManager
    log=begin(tmp_path)
    log.measurement(110., 1010., xy=(2.1,3.), value=.95, sample_count=20)
    calls=[]
    c=NS(hrs_run_log=log,hrs_log_source=(2.,3.),latest_pose=NS(pose=NS(position=NS(x=2.1,y=3.))),
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=120_000_000_000)),
         get_logger=lambda:NS(info=lambda msg:None,warning=lambda msg:None,error=lambda msg:None),
         hrs_response_threshold=.9,hrs_manager=HrsManager(),gmrf=NS(var_cells=[(0,0)]),
         history=NS(record_confirmed_event=lambda *args,**kwargs:True),current_event_id='event-1',
         publish_history=lambda:None,persist_history=lambda msg:True,repeat_after_hrs=True)
    def transition(reason):
        assert log.active is None
        row=next(csv.DictReader((log.directory/'runs.csv').open()))
        assert row['source_x']=='2.0' and row['outcome']=='detected'
        c.hrs_log_source=(99.,99.)
        calls.append('transition')
    c.start_source_transition=transition
    assert HrsWorkflow(c).confirm_hrs_response(0,.95,0.)
    assert calls==['transition']


def test_lrs_boundary_is_recorded_before_hrs_selection(tmp_path):
    from icir_cleanroom.gas_mapping.ros.lrs_workflow import LrsWorkflow
    from icir_cleanroom.gas_mapping.config import ControllerConfig
    log=HrsRunLog(tmp_path,'ros_simulation')
    c=NS(hrs_run_log=log,hrs_log_source=None,current_event_id='event-1',
         config=ControllerConfig.defaults(),lrs_lap=1,lap_hazard_detected=True,
         lap_max_concentration=.5,hazard_threshold=.2,
         get_clock=lambda:NS(now=lambda:NS(nanoseconds=100_000_000_000)),
         get_logger=lambda:NS(info=lambda msg:None,error=lambda msg:None),
         persist_history=lambda msg:None)
    calls=[]
    def planning():
        assert log.active['start_ros']==100.
        calls.append('plan')
    c.start_hrs_planning=planning
    LrsWorkflow(c).finish_lrs_navigation()
    assert calls==['plan']
