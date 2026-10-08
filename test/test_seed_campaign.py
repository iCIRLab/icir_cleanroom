"""Independent campaigns must never restore old inputs or mix retry outputs."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('seed_campaign', SCRIPTS / 'run_seed_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def package(tmp_path):
    p = tmp_path / 'package'
    (p / 'config/environments').mkdir(parents=True)
    (p / 'config/mapping').mkdir()
    poison = tmp_path / 'existing_history.json'
    poison.write_text('DO NOT LOAD THIS HISTORY')
    (p / 'config/environments/cleanroom_amc.yaml').write_text(yaml.safe_dump(dict(
        environment=dict(navigation_profile='config/navigation.yaml'),
        gas_environment=dict(source_random_seed=-1, persist_source_state=True), lrs={})))
    (p / 'config/mapping/default.yaml').write_text(yaml.safe_dump(dict(
        gas_mapping_controller_node=dict(ros__parameters=dict(
            load_history=True, save_history=True, history_file=str(poison),
            repeat_after_hrs=True, hrs_ucb_k=1, hrs_distance_weight=1)))))
    (p / 'config/nav2_params.yaml').write_text('{}\n')
    (p / 'config/navigation.yaml').write_text('{}\n')
    return p


def record(method='M7', k=.2, reason=campaign.NORMAL, x=1):
    return dict(method_id=method, schema_version='4', time_basis='ros_simulation',
                ros_clock_valid='True', termination_reason=reason,
                parameters=json.dumps(dict(hrs_ucb_k=k, hrs_distance_weight=0.0,
                    load_history=False, save_history=False, history_file='', repeat_after_hrs=False)),
                initial_seconds='2', search_seconds='3', total_seconds='5',
                hrs_distance_m='4', lrs_distance_m='6', total_distance_m='10',
                source_x=str(x), source_y='2', source_start_x=str(x), source_start_y='2',
                estimated_source_x=str(x), estimated_source_y='2', source_position_error='0',
                initial_update_completed='True', measurement_count='2', run_id='run', lrs_run_id='lap')


def store_result(folder, row):
    dest = folder / 'hrs/session'
    dest.mkdir(parents=True)
    tables = {
        'runs.csv': [row],
        'events.csv': [dict(event='lrs_complete_hrs_start', run_id='run', method_id=row['method_id'], details='{}'),
                       dict(event='hrs_end', run_id='run', method_id=row['method_id'],
                            details=json.dumps(dict(reason=row['termination_reason'])))],
        'estimate_history.csv': [dict(event='hrs_end', run_id='run', method_id=row['method_id'],
                                     elapsed_seconds=5, hrs_distance_m=4, source_position_error=0, ros_clock_valid=True)],
        'lrs_runs.csv': [dict(lrs_run_id='lap', lrs_distance_m=6, outcome='completed')],
    }
    for name, rows in tables.items():
        with (dest / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return dest / 'runs.csv'


@pytest.mark.parametrize('method,k,reason,expected', [
    ('M7', .2, campaign.NORMAL, 'normal'),
    ('M7', 0, campaign.NORMAL, 'configuration_mismatch'),
    ('M1', .2, campaign.NORMAL, 'configuration_mismatch'),
    ('M1', 0, 'node_shutdown', 'interrupted'),
    ('M2', 0, campaign.NORMAL, 'normal'),
    ('M7', .2, campaign.NORMAL + '; final approach stopped: status=6', 'final_approach_failed'),
    ('M1', 0, 'maximum 100 HRS target attempts reached', 'attempt_limit'),
])
def test_completion_weights_and_termination(method, k, reason, expected):
    assert campaign.validate(record(method, k, reason), method) == expected


@pytest.mark.parametrize('key,value', [('load_history', True), ('save_history', True),
                                      ('history_file', '/old/history.json'), ('repeat_after_hrs', True)])
def test_history_reuse_is_rejected_even_for_completed_rows(key, value):
    row = record()
    params = json.loads(row['parameters'])
    params[key] = value
    row['parameters'] = json.dumps(params)
    assert campaign.validate(row, 'M7') == 'history_policy_mismatch'


def test_bad_metrics_and_different_source():
    row = record()
    assert campaign.validate(row, 'M7', [2, 2]) == 'source_mismatch'
    row['total_seconds'] = 'nan'
    assert campaign.validate(row, 'M7') == 'invalid_metrics'
    row['total_seconds'] = '6'
    assert campaign.validate(row, 'M7') == 'inconsistent_metrics'


def test_prepare_350_fresh_configs_without_history_inputs(tmp_path, package):
    root = tmp_path / 'results'
    c = campaign.Campaign(root, package)
    c.prepare()
    assert len(c.state['tasks']) == 350
    assert sorted(p.name for p in root.glob('seed_*')) == [f'seed_{s}' for s in range(50, 100)]
    assert not list(root.rglob('*history.json'))
    assert not list(root.glob('*backup*'))
    for task in c.state['tasks']:
        folder = root / f'seed_{task["seed"]}' / task['method']
        cfg = yaml.safe_load((folder / 'run_config.yaml').read_text())
        assert cfg['controller']['load_history'] is False
        assert cfg['controller']['save_history'] is False
        assert cfg['controller']['history_file'] == ''
        assert cfg['controller']['hrs_ucb_k'] == campaign.expected_k(task['method'])
        assert cfg['profile']['gas_environment']['source_random_seed'] == task['seed']
    with (root / 'summary.csv').open(encoding='utf-8-sig') as stream:
        assert len(list(csv.DictReader(stream))) == 350
    c.prepare()
    assert len(c.state['tasks']) == 350


def test_no_importing_existing_seed_folders(tmp_path, package):
    root = tmp_path / 'results'
    (root / 'seed_50').mkdir(parents=True)
    with pytest.raises(ValueError, match='Existing seed folders'):
        campaign.Campaign(root, package).prepare()


@pytest.mark.parametrize('changes', [
    dict(policy='independent_lrs_hrs_no_history_v2'),
    dict(environment='aws_small_warehouse'),
    dict(environment=None),
])
def test_resume_rejects_legacy_environment_without_rewriting_results(tmp_path, package, changes):
    root = tmp_path / 'results'
    c = campaign.Campaign(root, package, 100, 100)
    c.prepare()
    c.state.update(changes)
    c.persist()
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='use a new results directory for cleanroom_amc'):
        campaign.Campaign(root, package, 100, 100).prepare()
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_campaign_uses_real_amc_world_robot_and_navigation(tmp_path):
    c = campaign.Campaign(tmp_path / 'amc', SCRIPTS.parent, 100, 100)
    c.prepare()
    assert c.state['environment'] == 'cleanroom_amc'
    manifest = json.loads((c.root / 'seed_100/manifest.json').read_text())
    assert manifest['environment'] == 'cleanroom_amc'
    for run in manifest['runs']:
        assert 'environment:=cleanroom_amc' in run['command']
        config = yaml.safe_load((c.root / 'seed_100' / run['method'] / 'run_config.yaml').read_text())
        env = config['profile']['environment']
        assert env['world'] == 'worlds/cleanroom_amc/cleanroom.world'
        assert env['map'] == 'maps/cleanroom_amc/cleanroom.yaml'
        assert env['robot']['entity'] == 'mobile_amc'
        assert env['robot']['model'] == 'urdf/mobile_amc.sdf'
        assert (env['robot_spawn']['x'], env['robot_spawn']['y']) == (-8.0, -5.0)
        nav = yaml.safe_load(Path(env['navigation_profile']).read_text())
        for key in ('local_costmap', 'global_costmap'):
            assert nav[key][key]['ros__parameters']['inflation_layer']['inflation_radius'] == .5
        gas = config['profile']['gas_environment']
        assert gas['source_mode'] == 'random_after_peak'
        assert gas['source_random_seed'] == 100
        assert gas['persist_source_state'] is False


def test_config_guard_rejects_accidental_history_file(tmp_path, package):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 50)
    c.prepare()
    task = c.state['tasks'][0]
    folder = c.root / 'seed_50/M1'
    manifest = json.loads((folder.parent / 'manifest.json').read_text())
    c.config_guard(task, manifest['runs'][0])
    (folder / 'initial_history.json').write_text('{}')
    with pytest.raises(ValueError, match='Unexpected history input'):
        c.config_guard(task, manifest['runs'][0])


def test_retry_replaces_current_output_and_does_not_retry_final_approach_failure(tmp_path, package, monkeypatch):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 50)
    c.prepare()
    calls = []

    def fake_run(method, folder, command, env, timeout):
        calls.append(timeout)
        assert not (folder / 'hrs').exists()
        reason = 'node_shutdown' if len(calls) == 1 else campaign.NORMAL + '; final approach stopped: status=6'
        row = record(method, campaign.expected_k(method), reason)
        path = store_result(folder, row)
        return dict(row, runner_status='completed', runs_csv=str(path)), False

    monkeypatch.setattr(campaign, 'run_one', fake_run)
    monkeypatch.setattr(campaign, 'free_port', lambda: 12345)
    c.run(limit=2)
    assert calls == [900, 1800]
    assert c.state['tasks'][0]['status'] == 'done'
    assert c.state['tasks'][0]['attempts'][0]['classification'] == 'interrupted'
    assert c.state['tasks'][0]['classification'] == 'final_approach_failed'
    assert c.state['status'] == 'paused'
    assert not list(c.root.glob('*backup*'))
    c.run(limit=0)
    assert len(calls) == 2


def test_actual_source_mismatch_stops_campaign(tmp_path, package, monkeypatch):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 50)
    c.prepare()

    def fake_run(method, folder, command, env, timeout):
        row = record(method, campaign.expected_k(method), x=1 if method == 'M1' else 2)
        path = store_result(folder, row)
        return dict(row, runner_status='completed', runs_csv=str(path)), False

    monkeypatch.setattr(campaign, 'run_one', fake_run)
    monkeypatch.setattr(campaign, 'free_port', lambda: 12345)
    c.run()
    assert c.state['status'] == 'stopped'
    assert c.state['tasks'][0]['status'] == 'done'
    assert c.state['tasks'][1]['classification'] == 'source_mismatch'
    assert c.state['tasks'][2]['attempts'] == []


@pytest.mark.parametrize('workers', [2, 3])
def test_parallel_overlap_isolation_resume_and_stop(tmp_path, package, monkeypatch, workers):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 49+workers)
    c.prepare()

    def fake_run(method, folder, command, env, timeout):
        (folder / 'started.json').write_text(json.dumps(dict(
            time=time.monotonic(), domain=env['ROS_DOMAIN_ID'], port=env['GAZEBO_MASTER_URI'])))
        # Wait for both lanes: this would fail with a serial implementation.
        deadline = time.monotonic() + 5
        while len(list(c.root.glob('seed_*/M1/started.json'))) < workers:
            assert time.monotonic() < deadline
            time.sleep(.02)
        row = record(method, campaign.expected_k(method))
        path = store_result(folder, row)
        (folder / 'ended.txt').write_text(str(time.monotonic()))
        return dict(row, runner_status='completed', runs_csv=str(path)), False

    monkeypatch.setattr(campaign, 'run_one', fake_run)
    c.run_parallel(limit=workers, workers=workers)
    assert c.state['status'] == 'paused'
    assert c.state['counts'] == {'done': workers, 'pending': 6*workers}
    starts = [json.loads(p.read_text()) for p in sorted(c.root.glob('seed_*/M1/started.json'))]
    ends = [float(p.read_text()) for p in c.root.glob('seed_*/M1/ended.txt')]
    assert max(s['time'] for s in starts) < min(ends)
    assert {s['domain'] for s in starts} == {str(60+i*10) for i in range(workers)}
    assert len({s['port'] for s in starts}) == workers
    saved = {p: p.read_bytes() for p in c.root.glob('seed_*/M1/hrs/*/*.csv')}
    (c.root / 'STOP').touch()
    c.run_parallel(workers=workers)
    assert c.state['counts'] == {'done': workers, 'pending': 6*workers}
    assert all(p.read_bytes() == content for p, content in saved.items())
    assert c.state['active'] == []


def test_parallel_stop_drains_both_running_lanes(tmp_path, package, monkeypatch):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 51)
    c.prepare()

    def fake_run(method, folder, command, env, timeout):
        time.sleep(.3)
        (c.root / 'STOP').touch()
        row = record(method, campaign.expected_k(method))
        path = store_result(folder, row)
        return dict(row, runner_status='completed', runs_csv=str(path)), False

    monkeypatch.setattr(campaign, 'run_one', fake_run)
    c.run_parallel()
    assert c.state['status'] == 'paused'
    assert c.state['counts'] == {'done': 2, 'pending': 12}
    assert all(not t['attempts'] for t in c.state['tasks'] if t['method'] != 'M1')


def test_v5_evaluation_and_timing_are_checked_alongside_legacy_rows(tmp_path):
    from icir_cleanroom.gas_mapping.application.benchmark import EvaluationReference
    row = record()
    assert campaign.validate(row, 'M7') == 'normal'  # Old schema remains readable.
    row.update(schema_version='5', **EvaluationReference((1., 2.), ((1., 2.),), False,
        'source_reachable', 0.).evaluate((1., 2.), (1., 2.)))
    row.update(lrs_seconds='10', cumulative_lrs_seconds='10', experiment_seconds='17',
               experiment_start_ros_seconds='0', experiment_ros_clock_valid='True',
               experiment_completed='True')
    # Serialize exactly as the CSV writer does, including booleans and nulls.
    row = {k: '' if v is None else str(v) for k,v in row.items()}
    assert campaign.validate(row, 'M7') == 'normal'
    row['adjusted_source_position_error'] = '1'
    assert campaign.validate(row, 'M7') != 'normal'
    row['adjusted_source_position_error'] = '0'
    row['experiment_seconds'] = '14'  # Less than LRS + HRS.
    assert campaign.validate(row, 'M7') != 'normal'


def test_v5_artifacts_reject_disagreement_with_final_history(tmp_path):
    from icir_cleanroom.gas_mapping.application.benchmark import EVALUATION_FIELDS, EvaluationReference
    row = record()
    row.update(schema_version='5', **EvaluationReference((1.,2.), ((1.,2.),), False,
        'source_reachable', 0.).evaluate((1.,2.), (1.,2.)))
    row = {k: '' if v is None else str(v) for k,v in row.items()}
    path = store_result(tmp_path, row)
    history_path = path.with_name('estimate_history.csv')
    history = list(csv.DictReader(history_path.open()))
    history[-1].update({k: row[k] for k in EVALUATION_FIELDS})
    def write_history():
        with history_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[-1]))
            writer.writeheader()
            writer.writerows(history)
    write_history()
    assert campaign.validate_artifacts(row, path)
    history[-1]['adjusted_source_position_error'] = '1'
    write_history()
    assert not campaign.validate_artifacts(row, path)


def test_cli_prepares_100_to_199_without_launching(tmp_path, package, monkeypatch):
    root = tmp_path / 'new_campaign'
    def forbidden(*args, **kwargs):
        pytest.fail('prepare-only must never launch an experiment')
    monkeypatch.setattr(campaign.Campaign, 'run', forbidden)
    monkeypatch.setattr(campaign.Campaign, 'run_parallel', forbidden)
    campaign.main(['--results', str(root), '--package', str(package),
                   '--start-seed', '100', '--end-seed', '199', '--workers', '2', '--prepare-only'])
    state = json.loads((root/'progress.json').read_text())
    assert state['start_seed'] == 100 and state['end_seed'] == 199
    assert len(state['tasks']) == 700
    assert state['counts'] == {'pending': 700}
    assert {t['seed'] for t in state['tasks']} == set(range(100, 200))
    assert not list(root.glob('seed_*/M*/hrs'))
    # The same explicit range resumes; the legacy default cannot import it.
    resumed = campaign.Campaign(root, package, 100, 199)
    resumed.prepare()
    with pytest.raises(ValueError, match='policy/environment/range'):
        campaign.Campaign(root, package).prepare()


@pytest.mark.parametrize('start,end', [('-1','199'), ('199','100')])
def test_cli_rejects_invalid_range_before_creating_output(tmp_path, start, end):
    root = tmp_path / 'invalid'
    with pytest.raises(SystemExit) as error:
        campaign.main(['--results', str(root), '--start-seed', start,
                       '--end-seed', end, '--prepare-only'])
    assert error.value.code == 2 and not root.exists()


def test_resume_two_workers_as_three_preserves_done_results(tmp_path, package, monkeypatch):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 53)
    c.prepare()
    def fake_run(method, folder, command, env, timeout):
        row = record(method, campaign.expected_k(method))
        path = store_result(folder, row)
        return dict(row, runner_status='completed', runs_csv=str(path)), False
    monkeypatch.setattr(campaign, 'run_one', fake_run)
    c.run_parallel(limit=2, workers=2)
    saved = {p:p.read_bytes() for p in c.root.glob('seed_*/M1/hrs/*/*.csv')}
    assert c.state['counts'] == {'done':2, 'pending':26}
    # Reload persisted state, as the user's next command will.
    resumed = campaign.Campaign(c.root, package, 50, 53)
    resumed.prepare()
    resumed.run_parallel(workers=3)
    assert resumed.state['status'] == 'complete'
    assert resumed.state['counts'] == {'done':28}
    assert all(len(t['attempts']) == 1 for t in resumed.state['tasks'])
    assert all(p.read_bytes() == content for p,content in saved.items())
    assert resumed.state['workers'] == 3


def test_marked_operator_stop_does_not_consume_retry_budget(tmp_path, package, monkeypatch):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 50)
    c.prepare()
    task = c.state['tasks'][0]
    task['attempts'] = [dict(number=1, retry_budget_exempt=True,
                             interruption_reason='user_requested_worker_change')]
    task['status'] = 'pending'
    def fake_run(method, folder, command, env, timeout):
        assert timeout == 900.0  # First charged attempt, despite one operator stop.
        row = record(method, campaign.expected_k(method))
        path = store_result(folder, row)
        return dict(row, runner_status='completed', runs_csv=str(path)), False
    monkeypatch.setattr(campaign, 'run_one', fake_run)
    c.run_parallel(limit=1, workers=3)
    assert task['status'] == 'done'
    assert len(task['attempts']) == 2 and campaign.attempts_used(task) == 1
    assert task['attempts'][-1]['number'] == 2


def test_three_worker_failures_still_get_only_one_automatic_retry(tmp_path, package, monkeypatch):
    c = campaign.Campaign(tmp_path / 'results', package, 50, 52)
    c.prepare()
    def fake_run(method, folder, command, env, timeout):
        return dict(method_id=method, runner_status='timeout'), False
    monkeypatch.setattr(campaign, 'run_one', fake_run)
    c.run_parallel(workers=3)
    assert c.state['status'] == 'incomplete'
    assert c.state['counts'] == {'failed':21}
    assert all([a['timeout'] for a in t['attempts']] == [900.,1800.]
               for t in c.state['tasks'])


def test_cli_routes_three_workers_without_launching_ros(tmp_path, package, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign.psutil, 'process_iter', lambda *a, **k: iter(()))
    monkeypatch.setattr(campaign.Campaign, 'run_parallel',
                        lambda self, limit=None, workers=2: calls.append((limit,workers)))
    campaign.main(['--results', str(tmp_path/'three'), '--package', str(package),
                   '--start-seed', '100', '--end-seed', '102', '--workers', '3', '--run'])
    assert calls == [(None,3)]
