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
        row = dict(method_id=method, outcome='failed' if method == 'M2' else 'detected',
                   termination_reason='test_limit' if method == 'M2' else 'test_threshold',
                   initial_seconds='2', search_seconds='3', total_seconds='5',
                   measurement_count='2', time_basis='fake_clock')
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
        assert_stopped(folder)
        rows.append(row)
    runner.save_summary(tmp_path, rows)
    saved = list(csv.DictReader((tmp_path/'summary.csv').open(encoding='utf-8-sig')))
    assert [r['method_id'] for r in saved] == list(runner.METHODS)
    assert saved[1]['outcome'] == 'failed'
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
