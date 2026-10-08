#!/usr/bin/env python3
"""Independent cleanroom_amc LRS-to-HRS experiments, M1..M7, no history.

Only results/seed_N/Mi holds current run data. No backup branches or old
results are imported. Retry causes and metrics remain in progress.json.
"""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_methods_once import FIELDS, METHODS, code_version, free_port, read_result, run_one

ENVIRONMENT = 'cleanroom_amc'
POLICY = 'independent_lrs_hrs_cleanroom_amc_no_history_v3'
NORMAL = 'no relative improvement in 10 consecutive HRS attempts'
LOG = logging.getLogger('seed_campaign')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_csv(path, fields, rows):
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def expected_k(method):
    return 0.2 if method == 'M7' else 0.0


def attempts_used(task):
    """Operator-marked interruptions remain in history without using the retry budget."""
    return sum(not attempt.get('retry_budget_exempt', False) for attempt in task['attempts'])


def usable(kind):
    return kind in ('normal', 'final_approach_failed', 'attempt_limit')


def attempt_process(connection, method, folder, command, env, timeout):
    """Only this child owns the ROS launch; the coordinator owns campaign files."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        connection.send(run_one(method, folder, command, env, timeout))
    finally:
        connection.close()


def validate(row, method, source=None):
    if row is None:
        return 'missing'
    try:
        params = json.loads(row['parameters'])
        if (params.get('load_history') is not False or params.get('save_history') is not False
                or params.get('history_file') != '' or params.get('repeat_after_hrs') is not False):
            return 'history_policy_mismatch'
        if params.get('hrs_ucb_k') != expected_k(method) or params.get('hrs_distance_weight') != 0.0:
            return 'configuration_mismatch'
        if row['method_id'] != method or row['schema_version'] not in ('4', '5', '6'):
            return 'invalid_identity'
        if row['ros_clock_valid'] != 'True' or row['time_basis'] != 'ros_simulation':
            return 'invalid_clock'
        if row['termination_reason'] == 'node_shutdown':
            return 'interrupted'
        reason = row['termination_reason']
        limits = (NORMAL, 'maximum 100 HRS target attempts reached')
        if not any(reason == limit or reason.startswith(limit + '; final approach stopped:') for limit in limits):
            return 'unexpected_termination'
        if row.get('initial_update_completed') != 'True' or int(row['measurement_count']) <= 0:
            return 'no_measurements'
        for key in ('initial_seconds', 'search_seconds', 'total_seconds',
                    'hrs_distance_m', 'lrs_distance_m', 'total_distance_m', 'source_position_error'):
            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                return 'invalid_metrics'
        xy = [float(row['source_x']), float(row['source_y'])]
        pairs = [
            (float(row['initial_seconds']) + float(row['search_seconds']), float(row['total_seconds'])),
            (float(row['hrs_distance_m']) + float(row['lrs_distance_m']), float(row['total_distance_m'])),
            (math.hypot(xy[0] - float(row['estimated_source_x']), xy[1] - float(row['estimated_source_y'])),
             float(row['source_position_error']))]
        pairs.extend((float(row['source_start_' + axis]), value) for axis, value in zip(('x', 'y'), xy))
        if any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7) for a, b in pairs):
            return 'inconsistent_metrics'
        if row['schema_version'] in ('5', '6'):
            from icir_cleanroom.gas_mapping.application.benchmark import validate_evaluation, validate_timing
            if not validate_evaluation(row) or not validate_timing(row):
                return 'inconsistent_evaluation'
        if source is not None and any(not math.isclose(a, b, abs_tol=1e-7) for a, b in zip(xy, source)):
            return 'source_mismatch'
    except (KeyError, TypeError, ValueError):
        return 'invalid_record'
    if 'final approach stopped:' in reason:
        return 'final_approach_failed'
    return 'attempt_limit' if reason.startswith('maximum ') else 'normal'


def validate_artifacts(raw, path):
    try:
        tables = {}
        for name in ('events.csv', 'estimate_history.csv', 'lrs_runs.csv'):
            text = path.with_name(name).read_text()
            rows = list(csv.DictReader(text.splitlines()))
            if not text.endswith('\n') or not rows or any(None in r or None in r.values() for r in rows):
                return False
            tables[name] = rows
        events, history = tables['events.csv'], tables['estimate_history.csv']
        if events[0]['event'] != 'lrs_complete_hrs_start' or events[-1]['event'] != 'hrs_end':
            return False
        if json.loads(events[-1]['details'])['reason'] != raw['termination_reason']:
            return False
        if not all(r['run_id'] == raw['run_id'] and r['method_id'] == raw['method_id'] for r in events + history):
            return False
        last = history[-1]
        if last['event'] != 'hrs_end' or not all(r['ros_clock_valid'] == 'True' for r in history):
            return False
        for key, raw_key in (('elapsed_seconds', 'total_seconds'), ('hrs_distance_m', 'hrs_distance_m'),
                             ('source_position_error', 'source_position_error')):
            if not math.isclose(float(last[key]), float(raw[raw_key]), rel_tol=1e-9, abs_tol=1e-7):
                return False
        if raw['schema_version'] in ('5', '6'):
            from icir_cleanroom.gas_mapping.application.benchmark import EVALUATION_FIELDS
            if any(last.get(key) != raw.get(key) for key in EVALUATION_FIELDS):
                return False
        lrs = [r for r in tables['lrs_runs.csv'] if r['lrs_run_id'] == raw['lrs_run_id']]
        return len(lrs) == 1 and lrs[0]['outcome'] == 'completed' and math.isclose(
            float(lrs[0]['lrs_distance_m']), float(raw['lrs_distance_m']), rel_tol=1e-9, abs_tol=1e-7)
    except (OSError, KeyError, TypeError, ValueError):
        return False


class Campaign:
    def __init__(self, root, package, start=50, end=99, methods=METHODS):
        self.root, self.package = root.resolve(), package.resolve()
        self.start, self.end = start, end
        self.methods = tuple(methods)
        self.state_path = self.root / 'progress.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    def persist(self):
        self.state['updated_utc'] = now()
        self.state['counts'] = dict(Counter(t['status'] for t in self.state['tasks']))
        write_json(self.state_path, self.state)

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state is None:
            if list(self.root.glob('seed_*')):
                raise ValueError('Existing seed folders without this experiment state; refusing to import or overwrite')
            cfg = self.package / 'config'
            profile = yaml.safe_load((cfg / 'environments' / f'{ENVIRONMENT}.yaml').read_text())
            controller = yaml.safe_load((cfg / 'mapping/default.yaml').read_text())['gas_mapping_controller_node']['ros__parameters']
            controller.update(load_history=False, save_history=False, history_file='', repeat_after_hrs=False,
                              hrs_ucb_k=0.0, hrs_distance_weight=0.0)
            self.state = dict(policy=POLICY, environment=ENVIRONMENT, start_seed=self.start, end_seed=self.end,
                              created_utc=now(), status='preparing', active=None,
                              code_version=code_version(self.package),
                              base_profile=profile, base_controller=controller,
                              nav2_text=(cfg / 'nav2_params.yaml').read_text(),
                              navigation_text=(self.package / profile['environment']['navigation_profile']).read_text(),
                              methods=list(self.methods),
                              prepared_seeds=[], expected_sources={}, max_attempts=2,
                              tasks=[dict(seed=s, method=m, status='pending', attempts=[], row={})
                                     for s in range(self.start, self.end + 1) for m in self.methods])
            self.persist()
        if (self.state.get('policy') != POLICY or self.state.get('environment') != ENVIRONMENT
                or self.state['start_seed'] != self.start
                or self.state['end_seed'] != self.end
                or tuple(self.state.get('methods', METHODS)) != self.methods):
            raise ValueError('Experiment policy/environment/range/methods does not match; '
                             'use a new results directory for cleanroom_amc')
        for seed in range(self.start, self.end + 1):
            if seed in self.state['prepared_seeds']:
                continue
            root = self.root / f'seed_{seed}'
            if list(root.glob('M*/hrs/*/runs.csv')):
                raise ValueError(f'Untracked raw results: {root}')
            snapshot = root / 'snapshot'
            snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot / 'nav2_params.yaml').write_text(self.state['nav2_text'])
            (snapshot / 'navigation_profile.yaml').write_text(self.state['navigation_text'])
            (snapshot / 'environment.yaml').write_text(
                yaml.safe_dump(self.state['base_profile'], sort_keys=False))
            (snapshot / 'controller.yaml').write_text(
                yaml.safe_dump(self.state['base_controller'], sort_keys=False))
            map_yaml = self.package / self.state['base_profile']['environment'].get('map', '')
            if map_yaml.is_file():
                shutil.copy2(map_yaml, snapshot / map_yaml.name)
                image = yaml.safe_load(map_yaml.read_text()).get('image')
                if image and (map_yaml.parent / image).is_file():
                    shutil.copy2(map_yaml.parent / image, snapshot / Path(image).name)
            profile = copy.deepcopy(self.state['base_profile'])
            profile['gas_environment'].update(source_mode='random_after_peak', source_random_seed=seed,
                                               persist_source_state=False, source_state_file='')
            profile['environment']['navigation_profile'] = str(snapshot / 'navigation_profile.yaml')
            runs = []
            for i, method in enumerate(self.methods):
                folder = root / method
                folder.mkdir(exist_ok=True)
                params = dict(self.state['base_controller'], method=method, hrs_ucb_k=expected_k(method),
                              hrs_results_directory=str(folder / 'hrs'))
                config = dict(profile=copy.deepcopy(profile), controller=params,
                              base_nav2_params=str(snapshot / 'nav2_params.yaml'))
                config_path = folder / 'run_config.yaml'
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                command = ['ros2', 'launch', 'icir_cleanroom', 'gas_mapping.launch.py',
                           f'environment:={ENVIRONMENT}', f'method:={method}',
                           'save_history:=false', 'repeat_after_hrs:=false', 'headless:=true',
                           f'run_config:={config_path}']
                runs.append(dict(method=method, ros_domain_id=40 + i, command=command,
                                 config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()))
            write_json(root / 'manifest.json', dict(seed=seed, policy=POLICY, environment=ENVIRONMENT,
                       source_mode='random_after_peak', load_history=False, save_history=False,
                       history_copied=False, gui=False,
                       code_version=self.state['code_version'], runs=runs))
            self.state['prepared_seeds'].append(seed)
            self.persist()
        if self.state['status'] == 'preparing':
            self.state['status'] = 'prepared'
        self.persist()
        self.summaries()

    def summaries(self):
        all_rows = []
        for seed in range(self.start, self.end + 1):
            rows = []
            for task in (t for t in self.state['tasks'] if t['seed'] == seed):
                row = dict(task.get('row', {}))
                row.setdefault('method_id', task['method'])
                row.setdefault('runner_status', 'not_started')
                if task['status'] == 'running':
                    row['runner_status'] = 'running'
                rows.append(row)
                all_rows.append(dict(row, seed=seed, classification=task.get('classification', ''),
                                     runner_attempts=len(task['attempts']), load_history=False,
                                     hrs_ucb_k=expected_k(task['method']), hrs_distance_weight=0.0))
            write_csv(self.root / f'seed_{seed}/summary.csv', FIELDS, rows)
        fields = ['seed', 'classification', 'runner_attempts', 'load_history', 'hrs_ucb_k', 'hrs_distance_weight'] + list(FIELDS)
        write_csv(self.root / 'summary.csv', fields, all_rows)

    def config_guard(self, task, run):
        folder = self.root / f'seed_{task["seed"]}' / task['method']
        path = folder / 'run_config.yaml'
        if hashlib.sha256(path.read_bytes()).hexdigest() != run['config_sha256']:
            raise ValueError(f'Prepared config changed: {path}')
        c = yaml.safe_load(path.read_text())
        p, g = c['controller'], c['profile']['gas_environment']
        if (p['load_history'] is not False or p['save_history'] is not False or p['history_file'] != ''
                or p['repeat_after_hrs'] is not False or p['hrs_ucb_k'] != expected_k(task['method'])
                or p['hrs_distance_weight'] != 0.0 or p['method'] != task['method']
                or g['source_random_seed'] != task['seed'] or g['persist_source_state'] is not False):
            raise ValueError(f'Invalid independent experiment config: {path}')
        if list(folder.glob('*history.json')) or (folder.parent / 'snapshot/history.json').exists():
            raise ValueError(f'Unexpected history input file: {folder}')

    def check_result(self, task):
        folder = self.root / f'seed_{task["seed"]}' / task['method']
        raw, path = read_result(folder / 'hrs', task['method'])
        kind = validate(raw, task['method'], self.state['expected_sources'].get(str(task['seed'])))
        if usable(kind) and not validate_artifacts(raw, path):
            kind = 'inconsistent_artifacts'
        return raw, path, kind

    def run_parallel(self, limit=None, workers=2):
        # Linux fork keeps the launch worker independent without initializing ROS
        # in the coordinator. Each lane has at most one live launch worker.
        if workers not in (2, 3):
            raise ValueError('parallel workers must be 2 or 3')
        context = multiprocessing.get_context('fork')
        jobs, launched = {}, 0
        labels = ('even', 'odd') if workers == 2 else tuple(f'worker_{i+1}' for i in range(workers))
        self.state.update(status='running', workers=workers, scheduling='seed_modulo', active=[],
                          pid=os.getpid(), pid_created=psutil.Process().create_time())
        for task in self.state['tasks']:
            raw, path, kind = self.check_result(task)
            if task['status'] == 'done' and not usable(kind):
                raise RuntimeError(f'Completed result changed: {task["seed"]}/{task["method"]}: {kind}')
            if task['status'] == 'running' and usable(kind):
                task.update(status='done', classification=kind,
                            row=dict(raw, runner_status='completed', runs_csv=str(path),
                                     events_csv=str(path.with_name('events.csv'))))
                self.state['expected_sources'].setdefault(str(task['seed']), [float(raw['source_x']), float(raw['source_y'])])
        self.persist()

        def save_active():
            self.state['active'] = [dict(worker=labels[lane],
                seed=j['task']['seed'], method=j['task']['method'], attempt=j['attempt']['number'],
                pid=j['process'].pid, ros_domain_id=j['attempt']['ros_domain_id'],
                gazebo_master_uri=j['attempt']['gazebo_master_uri']) for lane, j in sorted(jobs.items())]
            self.persist()
            self.summaries()

        def finish(job):
            result, stop = job['connection'].recv()
            job['process'].join(timeout=5)
            if job['process'].is_alive():
                raise RuntimeError('Launch worker did not exit after reporting cleanup')
            task, attempt = job['task'], job['attempt']
            raw, path, kind = self.check_result(task)
            attempt.update(ended_utc=now(), classification=kind, runner_result=result)
            completed = usable(kind) and result['runner_status'] == 'completed'
            task.update(row=result, classification=kind, status='done' if completed else 'failed')
            if completed:
                self.state['expected_sources'].setdefault(str(task['seed']), [float(raw['source_x']), float(raw['source_y'])])
            elif result['runner_status'] == 'completed':
                result['runner_status'] = 'invalid_result'
            job['connection'].close()
            LOG.info('END seed_%s/%s runner=%s classification=%s', task['seed'], task['method'], result['runner_status'], kind)
            return stop or kind in ('history_policy_mismatch', 'configuration_mismatch', 'source_mismatch')

        try:
            stopping = False
            while True:
                for job in jobs.values():
                    try:
                        for child in psutil.Process(job['process'].pid).children(recursive=True):
                            job['owned'][child.pid] = child
                    except psutil.NoSuchProcess:
                        pass
                for lane, job in list(jobs.items()):
                    if job['connection'].poll():
                        stopping = finish(job) or stopping
                        del jobs[lane]
                        save_active()
                    elif not job['process'].is_alive():
                        raise RuntimeError(f'Worker exited without a result: {job["process"].exitcode}')
                paused = (self.root / 'STOP').exists() or (limit is not None and launched >= limit)
                if not stopping and not paused:
                    for lane in range(workers):
                        if lane in jobs or (limit is not None and launched >= limit):
                            continue
                        task = next((t for t in self.state['tasks'] if t['seed'] % workers == lane
                                     and t['status'] != 'done' and attempts_used(t) < self.state['max_attempts']), None)
                        if task is None:
                            continue
                        folder = self.root / f'seed_{task["seed"]}' / task['method']
                        manifest = json.loads((folder.parent / 'manifest.json').read_text())
                        run = next(r for r in manifest['runs'] if r['method'] == task['method'])
                        self.config_guard(task, run)
                        if task['attempts']:
                            for name in ('hrs', 'ros_logs'):
                                if (folder / name).exists():
                                    shutil.rmtree(folder / name)
                            (folder / 'console.log').unlink(missing_ok=True)
                            task['row'] = {}
                        elif (folder / 'hrs').exists() or (folder / 'console.log').exists():
                            raise RuntimeError(f'Untracked output already exists: {folder}')
                        port = free_port()
                        while any(j['attempt']['gazebo_master_uri'] == f'http://127.0.0.1:{port}' for j in jobs.values()):
                            port = free_port()
                        domain = 60 + lane * 10
                        number = len(task['attempts']) + 1
                        timeout = 900.0 if attempts_used(task) == 0 else 1800.0
                        env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1',
                                   GAZEBO_MASTER_URI=f'http://127.0.0.1:{port}', ROS_LOG_DIR=str(folder / 'ros_logs'))
                        attempt = dict(number=number, started_utc=now(), timeout=timeout,
                                       worker=labels[lane], ros_domain_id=domain,
                                       gazebo_master_uri=env['GAZEBO_MASTER_URI'])
                        task['attempts'].append(attempt)
                        task['status'] = 'running'
                        # The manifest records the environment for the next/current run.
                        run.update(ros_domain_id=domain, worker=attempt['worker'], gazebo_master_uri=env['GAZEBO_MASTER_URI'])
                        write_json(folder.parent / 'manifest.json', manifest)
                        self.persist()  # Record ownership before creating any output.
                        receive, send = context.Pipe(duplex=False)
                        process = context.Process(target=attempt_process,
                            args=(send, task['method'], folder, run['command'], env, timeout),
                            name=f'campaign-{attempt["worker"]}')
                        process.start()
                        send.close()
                        jobs[lane] = dict(process=process, connection=receive, task=task, attempt=attempt, owned={})
                        launched += 1
                        save_active()
                        LOG.info('START worker=%s seed_%s/%s attempt=%s domain=%s gazebo=%s k=%s history=disabled',
                                 attempt['worker'], task['seed'], task['method'], number, domain, port, expected_k(task['method']))
                if not jobs:
                    self.state.update(status='stopped' if stopping else 'paused' if paused else
                                      'complete' if all(t['status'] == 'done' for t in self.state['tasks']) else 'incomplete')
                    break
                time.sleep(0.2)
        finally:
            # SIGINT lets run_one clean up its own process tree. Track descendants
            # as well, so an unexpectedly dead worker cannot leave a simulator.
            for job in jobs.values():
                if job['process'].is_alive():
                    os.kill(job['process'].pid, signal.SIGINT)
            for job in jobs.values():
                job['process'].join(timeout=35)
                survivors = []
                for p in job['owned'].values():
                    try:
                        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                            survivors.append(p)
                    except psutil.NoSuchProcess:
                        pass
                for p in survivors:
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(survivors, timeout=5)
                for p in alive:
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
                if job['process'].is_alive():
                    job['process'].kill()
                    job['process'].join(timeout=5)
                if job['connection'].poll():
                    try:
                        finish(job)
                    except (EOFError, OSError):
                        pass
                job['connection'].close()
            self.state['active'] = []
            if self.state['status'] == 'running':
                self.state['status'] = 'stopped'
            self.persist()
            self.summaries()

    def run(self, limit=None):
        self.state.update(status='running', pid=os.getpid(), pid_created=psutil.Process().create_time())
        self.persist()
        executed = 0
        for task in self.state['tasks']:
            raw, path, kind = self.check_result(task)
            if task['status'] == 'done':
                if not usable(kind):
                    raise RuntimeError(f'Completed result changed: {task["seed"]}/{task["method"]}: {kind}')
                continue
            # Recover a terminal result flushed just before the runner was interrupted.
            if task['status'] == 'running' and usable(kind):
                task.update(status='done', classification=kind)
                task['row'] = dict(raw, runner_status='completed', runs_csv=str(path),
                                   events_csv=str(path.with_name('events.csv')))
                self.state['expected_sources'].setdefault(str(task['seed']), [float(raw['source_x']), float(raw['source_y'])])
                self.persist()
                self.summaries()
                continue
            while attempts_used(task) < self.state['max_attempts']:
                if (self.root / 'STOP').exists() or (limit is not None and executed >= limit):
                    self.state.update(status='paused', active=None)
                    self.persist()
                    return
                folder = self.root / f'seed_{task["seed"]}' / task['method']
                manifest = json.loads((folder.parent / 'manifest.json').read_text())
                run = next(r for r in manifest['runs'] if r['method'] == task['method'])
                self.config_guard(task, run)
                if task['attempts']:
                    # User explicitly requested one current record, without backups.
                    for name in ('hrs', 'ros_logs'):
                        if (folder / name).exists():
                            shutil.rmtree(folder / name)
                    (folder / 'console.log').unlink(missing_ok=True)
                    task['row'] = {}
                elif (folder / 'hrs').exists() or (folder / 'console.log').exists():
                    raise RuntimeError(f'Untracked output already exists: {folder}')
                number = len(task['attempts']) + 1
                timeout = 900.0 if attempts_used(task) == 0 else 1800.0
                env = dict(os.environ, ROS_DOMAIN_ID=str(run['ros_domain_id']), ROS_LOCALHOST_ONLY='1',
                           GAZEBO_MASTER_URI=f'http://127.0.0.1:{free_port()}', ROS_LOG_DIR=str(folder / 'ros_logs'))
                attempt = dict(number=number, started_utc=now(), timeout=timeout)
                task['attempts'].append(attempt)
                task['status'] = 'running'
                self.state['active'] = dict(seed=task['seed'], method=task['method'], attempt=number)
                self.persist()
                self.summaries()
                LOG.info('START seed_%s/%s attempt=%s k=%s history=disabled', task['seed'], task['method'], number, expected_k(task['method']))
                result, stop = run_one(task['method'], folder, run['command'], env, timeout)
                executed += 1
                raw, path, kind = self.check_result(task)
                attempt.update(ended_utc=now(), classification=kind, runner_result=result)
                task.update(row=result, classification=kind)
                completed = usable(kind) and result['runner_status'] == 'completed'
                task['status'] = 'done' if completed else 'failed'
                if completed:
                    self.state['expected_sources'].setdefault(str(task['seed']), [float(raw['source_x']), float(raw['source_y'])])
                elif result['runner_status'] == 'completed':
                    task['row']['runner_status'] = 'invalid_result'
                self.state['active'] = None
                self.persist()
                self.summaries()
                LOG.info('END seed_%s/%s runner=%s classification=%s done=%s/%s', task['seed'], task['method'],
                         result['runner_status'], kind, sum(t['status'] == 'done' for t in self.state['tasks']), len(self.state['tasks']))
                if stop or kind in ('history_policy_mismatch', 'configuration_mismatch', 'source_mismatch'):
                    self.state['status'] = 'stopped'
                    self.persist()
                    return
                if completed:
                    break
        self.state.update(status='complete' if all(t['status'] == 'done' for t in self.state['tasks']) else 'incomplete', active=None)
        self.persist()
        self.summaries()
        LOG.info('Campaign %s', self.state['status'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--start-seed', type=int, default=50, help='First seed, inclusive (default: 50)')
    parser.add_argument('--end-seed', type=int, default=99, help='Last seed, inclusive (default: 99)')
    parser.add_argument('--methods', default=','.join(METHODS),
                        help=f'Comma-separated subset of {",".join(METHODS)} (default: all)')
    args = parser.parse_args(argv)
    args.methods = tuple(m.strip().upper() for m in args.methods.split(',') if m.strip())
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown or not args.methods:
        parser.error(f'--methods must be a non-empty subset of {",".join(METHODS)}')
    if args.start_seed < 0 or args.end_seed < args.start_seed:
        parser.error('seed range must satisfy 0 <= start-seed <= end-seed')
    args.results = args.results.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.results / 'runner.log')])
    with (args.results / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.run:
            active = []
            for p in psutil.process_iter(['pid', 'name', 'status']):
                if p.info['name'] in ('gzserver', 'gzclient') and p.info['status'] != psutil.STATUS_ZOMBIE:
                    active.append(p.pid)
            if active:
                raise RuntimeError(f'Gazebo already running; no concurrent simulation: {active}')
        campaign = Campaign(args.results, args.package, args.start_seed, args.end_seed, args.methods)
        campaign.prepare()
        LOG.info('Prepared %s independent runs on %s, seeds %s..%s, history disabled',
                 len(campaign.state['tasks']), ENVIRONMENT, args.start_seed, args.end_seed)
        if args.run:
            try:
                if args.workers > 1:
                    campaign.run_parallel(args.limit, workers=args.workers)
                else:
                    campaign.run(args.limit)
            except BaseException:
                campaign.state.update(status='stopped', active=None)
                campaign.persist()
                raise


if __name__ == '__main__':
    main()
