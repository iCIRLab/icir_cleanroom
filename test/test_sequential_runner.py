"""Exercise the real subprocess runner against fake ROS processes, never Gazebo."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import psutil
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sequential_runner', ROOT/'scripts/run_methods_once.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def fake_launch(tmp_path):
    script = tmp_path/'fake_launch.py'
    script.write_text('''import csv, pathlib, sys, time, subprocess, os
folder, method, mode = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
(folder/'child.pid').write_text(str(child.pid))
try:
    time.sleep(.15)
    if mode == 'exit':
        sys.exit(3)
    if mode == 'result':
        path = folder/'hrs'/'session'/'runs.csv'
        path.parent.mkdir(parents=True)
        row = dict(method_id=method, schema_version=4,
                   termination_reason='test_limit' if method == 'M2' else 'test_threshold',
                   initial_seconds='2', search_seconds='3', total_seconds='5',
                   measurement_count='2', time_basis='fake_clock', total_distance_m='32.5',
                   estimated_source_x='2.0', estimated_source_y='3.0', source_position_error='0.5',
                   lrs_distance_m='20.0', hrs_distance_m='12.5', lrs_run_id='lap-1')
        with path.open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    while True: time.sleep(.1)
except KeyboardInterrupt:
    pass
finally:
    child.terminate()
    child.wait()
''')
    return script


def assert_stopped(folder):
    pid = int((folder/'child.pid').read_text())
    try:
        assert not psutil.Process(pid).is_running() or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_seven_methods_real_subprocess_sequence_and_summary(tmp_path, fake_launch):
    rows = []
    for method in runner.METHODS:
        folder = tmp_path/method
        folder.mkdir()
        row, stop = runner.run_one(method, folder,
            [sys.executable, str(fake_launch), str(folder), method, 'result'],
            os.environ.copy(), timeout=5., poll=.03)
        assert row['runner_status'] == 'completed' and not stop
        assert row['method_id'] == method
        assert row['total_seconds'] == '5'
        assert row['total_distance_m'] == '32.5'
        assert_stopped(folder)
        rows.append(row)
    runner.save_summary(tmp_path, rows)
    saved = list(csv.DictReader((tmp_path/'summary.csv').open(encoding='utf-8-sig')))
    assert [r['method_id'] for r in saved] == list(runner.METHODS)
    assert 'outcome' not in saved[1]
    assert all(r['source_position_error'] == '0.5' for r in saved)
    assert all(r['schema_version'] == '4' for r in saved)
    assert all(row['total_distance_m'] == '32.5' for row in saved)
    assert all(row['lrs_distance_m'] == '20.0' and row['hrs_distance_m'] == '12.5'
               and row['lrs_run_id'] == 'lap-1' for row in saved)
    markdown = (tmp_path/'summary.md').read_text()
    assert 'LRS Distance (m)' in markdown and 'HRS Distance D (m)' in markdown
    assert '| 20.0 | 12.5 | 32.5 |' in markdown
    assert len(list(tmp_path.glob('M*/hrs/session/runs.csv'))) == 7


@pytest.mark.parametrize('mode,status', [('timeout','timeout'), ('exit','process_exited')])
def test_runner_failure_cleanup_and_missing_times(tmp_path, fake_launch, mode, status):
    folder = tmp_path/'M1'
    folder.mkdir()
    row, stop = runner.run_one('M1', folder,
        [sys.executable, str(fake_launch), str(folder), 'M1', mode],
        os.environ.copy(), timeout=.5, poll=.03)
    assert row['runner_status'] == status
    assert row.get('total_seconds', '') == '' and not stop
    assert_stopped(folder)


def test_summary_recomputes_legacy_totals_and_does_not_invent_missing_lrs(tmp_path):
    rows = [
        dict(method_id='M1', schema_version='2', total_distance_m='5.0'),
        dict(method_id='M2', schema_version='2', total_distance_m='5.0',
             lrs_distance_m='10.0', hrs_distance_m='5.0'),
        dict(method_id='M3', schema_version='3', total_distance_m='99.0',
             lrs_distance_m='0.0', hrs_distance_m='0.0'),
        dict(method_id='M4', schema_version='3', total_distance_m='',
             lrs_distance_m='10.0', hrs_distance_m=''),
    ]
    runner.save_summary(tmp_path, rows)
    with (tmp_path/'summary.csv').open(encoding='utf-8-sig') as stream:
        saved = list(csv.DictReader(stream))
    assert [r['total_distance_m'] for r in saved] == ['', '15.0', '0.0', '']
    assert saved[0]['hrs_distance_m'] == '5.0'
    assert rows[0]['total_distance_m'] == '5.0'  # Raw legacy results stay intact.


def test_completed_round_resume_preserves_old_distances_and_new_schema(tmp_path):
    raw_rows = [
        dict(method_id='M1', schema_version='2', total_distance_m='5.0'),
        dict(method_id='M2', schema_version='2', total_distance_m='5.0',
             lrs_distance_m='10.0', hrs_distance_m='5.0'),
        dict(method_id='M3', schema_version='3', total_distance_m='15.0',
             lrs_distance_m='10.0', hrs_distance_m='5.0'),
        dict(method_id='M4', schema_version='3', total_distance_m='99.0',
             lrs_distance_m='10.0'),  # Missing HRS must not use a v3 total.
        dict(method_id='M5', schema_version='1'),  # Before distance logging.
    ]
    originals = {}
    for row in raw_rows:
        row.update(outcome='failed', termination_reason='test')
        path = tmp_path / row['method_id'] / 'hrs' / 'session' / 'runs.csv'
        path.parent.mkdir(parents=True)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        originals[path] = path.read_bytes()
    (tmp_path / 'manifest.json').write_text(json.dumps(
        dict(runs=[dict(method=row['method_id']) for row in raw_rows])))
    # All methods completed: use the real resume CLI without launching ROS.
    result = runner.subprocess.run(
        [sys.executable, str(ROOT / 'scripts/resume_round.py'), str(tmp_path)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    with (tmp_path / 'summary.csv').open(encoding='utf-8-sig') as stream:
        saved = list(csv.DictReader(stream))
    assert [row['hrs_distance_m'] for row in saved] == ['5.0', '5.0', '5.0', '', '']
    assert [row['total_distance_m'] for row in saved] == ['', '15.0', '15.0', '', '']
    assert [row['schema_version'] for row in saved] == ['2', '2', '3', '3', '1']
    for path, original in originals.items():
        assert path.read_bytes() == original


def test_reader_rejects_partial_mismatched_and_repeated_results(tmp_path):
    path = tmp_path/'session'/'runs.csv'
    path.parent.mkdir()
    path.write_text('method_id,outcome,termination_reason\nM1,detected')
    assert runner.read_result(tmp_path,'M1') == (None,None)
    path.write_text('method_id,outcome,termination_reason\nM2,detected,test\n')
    with pytest.raises(ValueError):
        runner.read_result(tmp_path,'M1')
    path.write_text('method_id,outcome,termination_reason\nM1,detected,test\nM1,failed,test\n')
    with pytest.raises(ValueError):
        runner.read_result(tmp_path,'M1')


def test_history_path_is_preserved_in_summary_without_requiring_old_runs_to_have_it(tmp_path):
    session = tmp_path / 'M1' / 'hrs' / 'session'
    session.mkdir(parents=True)
    (session / 'runs.csv').write_text('method_id,termination_reason\nM1,done\n')
    row, _ = runner.read_result(session.parent, 'M1')
    assert row['estimate_history_csv'] == ''
    history = session / 'estimate_history.csv'
    history.write_text('run_id,elapsed_seconds,source_position_error\nr1,10,0.5\n')
    row, _ = runner.read_result(session.parent, 'M1')
    runner.save_summary(tmp_path, [row])
    with (tmp_path / 'summary.csv').open(encoding='utf-8-sig') as stream:
        saved = next(csv.DictReader(stream))
    assert saved['estimate_history_csv'] == str(history)
    assert 'estimate_history.csv' in (tmp_path / 'summary.md').read_text()


def test_prepare_freezes_history_configs_and_uses_independent_directories(tmp_path):
    package = tmp_path/'package'
    (package/'config/environments').mkdir(parents=True)
    (package/'config/mapping').mkdir()
    history = tmp_path/'original.json'
    history.write_text('{"test": 1}')
    profile = dict(environment={'navigation_profile':'nav.yaml'}, gas_environment={}, lrs={})
    (package/'config/environments/warehouse.yaml').write_text(yaml.safe_dump(profile))
    (package/'config/mapping/default.yaml').write_text(yaml.safe_dump(
        {'gas_mapping_controller_node':{'ros__parameters':{'history_file':str(history)}}}))
    (package/'nav.yaml').write_text('{}')
    (package/'config/nav2_params.yaml').write_text('{}')
    output = tmp_path/'output'
    manifest = runner.prepare(output, package, 'warehouse', 8, 80, False, False)
    for i, method in enumerate(runner.METHODS):
        config = yaml.safe_load((output/method/'run_config.yaml').read_text())
        assert config['controller']['save_history'] is False
        assert config['controller']['repeat_after_hrs'] is False
        assert config['controller']['method'] == method
        assert config['profile']['gas_environment']['source_random_seed'] == 8
        assert config['profile']['gas_environment']['persist_source_state'] is False
        assert Path(config['controller']['history_file']).read_bytes() == history.read_bytes()
        assert str(output/method) in config['controller']['hrs_results_directory']
        assert manifest['runs'][i]['ros_domain_id'] == 80+i
    assert history.read_text() == '{"test": 1}'
    with pytest.raises(FileExistsError):
        runner.prepare(output, package, 'warehouse', 8, 80, False, False)


def test_lrs_only_interruption_preserves_partial_times_in_summary(tmp_path):
    from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
    log = HrsRunLog(tmp_path/'hrs', 'ros_simulation')
    log.start_lrs(10, 100, event_id='e', lrs_lap=1, method_id='M1')
    log.finish_lrs(30, 125, outcome='interrupted', reason='node_shutdown')
    row = dict(method_id='M1', runner_status='interrupted',
               **runner.read_lrs_result(tmp_path/'hrs', 'M1'))
    runner.save_summary(tmp_path, [row])
    saved = next(csv.DictReader((tmp_path/'summary.csv').open(encoding='utf-8-sig')))
    assert saved['lrs_seconds'] == '20'
    assert saved['experiment_seconds'] == '20'
    assert saved['wall_experiment_seconds'] == '25'
    assert saved['experiment_completed'] == 'False'
    assert saved['total_seconds'] == '' and saved['schema_version'] == ''
    assert saved['lrs_schema_version'] == '2'
