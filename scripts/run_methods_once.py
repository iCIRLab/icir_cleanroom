#!/usr/bin/env python3
"""Run M1..M7 once sequentially. This program launches ROS; importing it does not."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time

import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icir_cleanroom.gas_mapping.application.benchmark import (
    EVALUATION_FIELDS, SOURCE_TRUTH_FIELDS, TIME_FIELDS)


METHODS = tuple(f'M{i}' for i in range(1, 8))
FIELDS = ('method_id', 'runner_status', 'termination_reason', 'schema_version',
          'time_basis', 'initial_seconds', 'search_seconds', 'total_seconds',
          'total_distance_m', 'lrs_distance_m', 'hrs_distance_m', 'lrs_run_id',
          'wall_initial_seconds', 'wall_search_seconds', 'wall_total_seconds',
          'measurement_count', 'attempt_count', 'search_iterations',
          'source_x', 'source_y', 'final_robot_x', 'final_robot_y',
          'estimated_source_x', 'estimated_source_y', 'source_position_error',
          'final_measured_concentration',
          'runner_elapsed_seconds', 'runs_csv', 'events_csv', 'estimate_history_csv',
          'console_log', 'note', 'lrs_runs_csv', 'lrs_schema_version',
          *SOURCE_TRUTH_FIELDS, *EVALUATION_FIELDS, *TIME_FIELDS)


def read_result(directory, method):
    """Read only fully written rows from this method's fresh output directory."""
    results = []
    for path in directory.glob('*/runs.csv'):
        content = path.read_text(encoding='utf-8')
        if not content.endswith('\n'):
            continue
        for row in csv.DictReader(io.StringIO(content)):
            if None in row or any(value is None for value in row.values()):
                continue
            if row.get('method_id') != method:
                raise ValueError(f'Unexpected method in {path}')
            if row.get('termination_reason'):
                history_path = path.with_name('estimate_history.csv')
                row['estimate_history_csv'] = str(history_path) if history_path.is_file() else ''
                results.append((row, path))
    if len(results) > 1:
        raise ValueError(f'More than one HRS run recorded for {method}')
    return results[0] if results else (None, None)



def read_lrs_result(directory, method):
    """Preserve recorded partial LRS timing when the run never reached HRS."""
    latest = {}
    for path in sorted(directory.glob('*/lrs_runs.csv')):
        content = path.read_text(encoding='utf-8')
        if not content.endswith('\n'):
            continue
        for row in csv.DictReader(io.StringIO(content)):
            if None in row or any(value is None for value in row.values()):
                continue
            if row.get('method_id') != method:
                raise ValueError(f'Unexpected method in {path}')
            latest = {key: row.get(key, '') for key in TIME_FIELDS}
            latest.update(lrs_runs_csv=str(path), lrs_schema_version=row.get('schema_version', ''),
                          lrs_run_id=row.get('lrs_run_id', ''),
                          lrs_distance_m=row.get('lrs_distance_m', ''),
                          experiment_completed=False)
    return latest


def save_summary(root, rows):
    # Old runs used total_distance_m for HRS alone. Recompute summaries from
    # the two components so resumed experiments cannot mix these meanings.
    rows = [dict(row) for row in rows]
    for row in rows:
        source_schema = str(row.get('schema_version') or 2)
        if row.get('hrs_distance_m') in (None, '') and source_schema in ('1', '2'):
            row['hrs_distance_m'] = row.get('total_distance_m', '')
        try:
            distances = [float(row.get(key, ''))
                         for key in ('lrs_distance_m', 'hrs_distance_m')]
            total_distance = sum(distances)
            row['total_distance_m'] = (total_distance if
                all(math.isfinite(value) and value >= 0 for value in distances)
                and math.isfinite(total_distance) else '')
        except (TypeError, ValueError):
            row['total_distance_m'] = ''
        # Keep the raw version: v4 includes the final approach in HRS T/D.
        # Existing records cannot acquire that missing travel by conversion.
    temporary = root/'summary.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, '') for key in FIELDS} for row in rows)
    temporary.replace(root/'summary.csv')
    lines = ['# M1–M7 results', '',
             'Each method runs once. HRS time starts at the LRS-to-HRS transition and includes the final move to the estimated source (schema v4/v5).',
             'Runner elapsed time includes startup, LRS and cleanup. No source success/failure classification is made.', '',
             '| Method | Runner | Schema | Initial (s) | Search (s) | HRS Time T (s) | LRS Distance (m) | HRS Distance D (m) | Total Distance (m) | Source Estimate Error E (m) | Adjusted Error (m) | LRS Time (s) | Experiment Time (s) | Reason |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    def cell(value):
        return str(value).replace('|', '/').replace('\n', ' ') if value != '' else '—'
    for row in rows:
        keys = ('method_id', 'runner_status', 'schema_version', 'initial_seconds',
                'search_seconds', 'total_seconds', 'lrs_distance_m', 'hrs_distance_m',
                'total_distance_m', 'source_position_error', 'adjusted_source_position_error',
                'cumulative_lrs_seconds', 'experiment_seconds', 'termination_reason')
        lines.append('| ' + ' | '.join(cell(row.get(key, '')) for key in keys) + ' |')
    lines += ['', 'Raw records: `M1/hrs/*/runs.csv`, `lrs_runs.csv`, `events.csv`, `estimate_history.csv`; similarly for M2–M7.',
              '`estimate_history.csv` records elapsed HRS time, cumulative HRS distance, and the best estimate/error so far at each measurement and through final arrival.',
              'LRS distance is the preceding patrol lap; every LRS lap is saved in `lrs_runs.csv`. In v4, HRS time and distance include the final approach.',
              'Total distance is LRS + HRS; it stays blank if either distance is missing. total_seconds remains HRS-only; experiment_seconds spans first LRS start through HRS end, including transitions.',
              'Error is the distance from the final estimated source to ground truth, not from the final robot pose. Older records retain their original schema and measurement boundaries.',
              'Process output: `M1/console.log`, etc. `manifest.json` and `snapshot/` preserve settings.',
              'Missing values remain blank. No normalization, scores, or averages are computed.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


class OwnedLaunch:
    """Track descendants by psutil identity; never kill unrelated ROS processes."""
    def __init__(self, command, env, output):
        self.process = subprocess.Popen(command, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        self.root = psutil.Process(self.process.pid)
        self.owned = {self.root}

    def track(self):
        for process in list(self.owned):
            try:
                self.owned.update(process.children(recursive=True))
            except psutil.NoSuchProcess:
                pass

    def live(self):
        self.process.poll()  # reap launch parent if it has exited
        live = []
        for process in self.owned:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process)
            except psutil.NoSuchProcess:
                pass
        return live

    def stop(self):
        # Let ROS launch terminate its children gracefully and flush CSV first.
        for sig, grace in ((signal.SIGINT, 15.), (signal.SIGTERM, 5.), (signal.SIGKILL, 5.)):
            self.track()
            targets = [self.root] if sig == signal.SIGINT and self.root.is_running() else self.live()
            for process in targets:
                try:
                    process.send_signal(sig)
                except psutil.NoSuchProcess:
                    pass
            deadline = time.monotonic()+grace
            while time.monotonic() < deadline:
                self.track()
                if not self.live():
                    self.process.wait()
                    return True
                time.sleep(.1)
        return not self.live()


def run_one(method, folder, command, env, timeout, poll=.5):
    result = {'method_id': method, 'runner_status': 'running',
              'console_log': str(folder/'console.log')}
    start = time.monotonic()
    launch = None
    interrupted = False
    with (folder/'console.log').open('w', encoding='utf-8') as output:
        try:
            launch = OwnedLaunch(command, env, output)
            while True:
                launch.track()
                row, _ = read_result(folder/'hrs', method)
                if row:
                    result['runner_status'] = 'completed'
                    break
                if launch.process.poll() is not None:
                    result.update(runner_status='process_exited',
                                  note=f'launch exit code {launch.process.returncode}')
                    break
                if time.monotonic()-start >= timeout:
                    result.update(runner_status='timeout', note='Full run wall-clock limit reached')
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            interrupted = True
            result['runner_status'] = 'interrupted'
        except Exception as error:
            result.update(runner_status='runner_error', note=str(error))
        finally:
            if launch is not None:
                # A second Ctrl+C must not leave this run's robots running.
                previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    if not launch.stop():
                        result.update(runner_status='cleanup_failed', note='Owned processes still running; batch stopped')
                except Exception as error:
                    result.update(runner_status='cleanup_failed', note=str(error))
                finally:
                    signal.signal(signal.SIGINT, previous)
    result['runner_elapsed_seconds'] = time.monotonic()-start
    try:
        row, path = read_result(folder/'hrs', method)
        if row:
            result.update({key: value for key, value in row.items() if key in FIELDS})
            result.update(runs_csv=str(path), events_csv=str(path.with_name('events.csv')))
        else:
            result.update(read_lrs_result(folder/'hrs', method))
            if result['runner_status'] == 'completed':
                result.update(runner_status='runner_error', note='Completed record disappeared')
    except (OSError, ValueError) as error:
        if result['runner_status'] != 'cleanup_failed':
            result.update(runner_status='runner_error', note=str(error))
    return result, interrupted or result['runner_status'] == 'cleanup_failed'


def code_version(package=None):
    """Identify the code a run was produced by, or say why it could not be.

    Resolves through the install-space symlink to the source tree, because the
    share directory itself is not a git working tree.
    """
    package = Path(__file__).resolve().parents[1] if package is None else package
    def git(*args):
        return subprocess.run(('git', '-C', str(package)) + args,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        commit = git('rev-parse', 'HEAD')
        if not commit:
            return {'available': False, 'reason': 'not a git working tree'}
        status = git('status', '--porcelain')
        return {'available': True, 'commit': commit,
                'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
                'describe': git('describe', '--always', '--dirty'),
                'dirty': bool(status),
                'dirty_paths': sorted(
                    line[3:] for line in status.splitlines())[:50]}
    except (OSError, subprocess.SubprocessError) as error:
        return {'available': False, 'reason': str(error)}


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def prepare(root, package, environment, seed, domain, gui, empty_history):
    """Freeze profiles/history without writing the user's existing settings."""
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root/'snapshot'
    snapshot.mkdir()
    profile = yaml.safe_load((package/'config/environments'/f'{environment}.yaml').read_text())
    controller = yaml.safe_load((package/'config/mapping/default.yaml').read_text())['gas_mapping_controller_node']['ros__parameters']
    original_history = Path(controller.get('history_file', '~/.ros/icir_cleanroom/gas_history.json')).expanduser()
    initial_history = snapshot/'history.json'
    copied = not empty_history and original_history.is_file()
    if copied:
        shutil.copy2(original_history, initial_history)
    # Same configured source parameters and RNG seed, with no persisted-source override.
    profile['gas_environment'].update(source_random_seed=seed, persist_source_state=False)
    controller.update(save_history=False, repeat_after_hrs=False)
    for name, content in [('environment.yaml', profile), ('mapping.yaml', controller)]:
        (snapshot/name).write_text(yaml.safe_dump(content, sort_keys=False))
    # Preserve human-readable navigation settings as well as the exact source paths.
    shutil.copy2(package/'config/nav2_params.yaml', snapshot/'nav2_params.yaml')
    shutil.copy2(package/profile['environment']['navigation_profile'], snapshot/'navigation_profile.yaml')
    profile['environment']['navigation_profile'] = str(snapshot/'navigation_profile.yaml')
    manifest = dict(environment=environment, methods=METHODS, seed=seed,
                    ros_domain_base=domain, gui=gui, package=str(package),
                    initial_history=str(original_history), history_copied=copied,
                    save_history=False, repeat_after_hrs=False,
                    note='Same starting config/history; live navigation and LRS measurements are not deterministic replays.',
                    runs=[])
    for index, method in enumerate(METHODS):
        folder = root/method
        folder.mkdir()
        if copied:
            shutil.copy2(initial_history, folder/'initial_history.json')
        params = dict(controller, method=method, history_file=str(folder/'initial_history.json'),
                      hrs_results_directory=str(folder/'hrs'))
        run_config = folder/'run_config.yaml'
        run_config.write_text(yaml.safe_dump(dict(
            profile=copy.deepcopy(profile), controller=params,
            base_nav2_params=str(snapshot/'nav2_params.yaml')), sort_keys=False))
        command = ['ros2', 'launch', 'icir_cleanroom', 'gas_mapping.launch.py',
                   f'environment:={environment}', f'method:={method}',
                   'repeat_after_hrs:=false', 'save_history:=false',
                   f'headless:={str(not gui).lower()}', f'run_config:={run_config}']
        manifest['runs'].append(dict(method=method, command=command, ros_domain_id=domain+index))
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', default='cleanroom_amc')
    parser.add_argument('--output', type=Path, help='New output directory; existing directory is rejected')
    parser.add_argument('--timeout', type=float, default=1800., help='Maximum wall seconds per method, including startup and LRS (default 1800)')
    parser.add_argument('--seed', type=int, default=0, help='Common source RNG seed (manual source stays at configured position)')
    parser.add_argument('--ros-domain-id', type=int, default=None, help='First of seven private ROS domains (0..94)')
    parser.add_argument('--gui', action='store_true', help='Show Gazebo and RViz; default headless')
    parser.add_argument('--empty-history', action='store_true', help='Start all methods without existing history; original file stays untouched')
    parser.add_argument('--dry-run', action='store_true', help='Prepare configs and summaries, print commands; start no processes')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.seed < 0:
        parser.error('timeout must be positive and finite, seed must be nonnegative')
    if args.ros_domain_id is not None and not 0 <= args.ros_domain_id <= 94:
        parser.error('ros-domain-id must be in 0..94')
    from ament_index_python.packages import get_package_share_directory
    package = Path(get_package_share_directory('icir_cleanroom'))
    domain = args.ros_domain_id if args.ros_domain_id is not None else random.SystemRandom().randrange(20, 90)
    root = (args.output or Path.cwd()/'results'/datetime.now(timezone.utc).strftime('methods_%Y%m%dT%H%M%S_%fZ')).expanduser().resolve()
    if "LaunchConfiguration('run_config')" not in (package/'launch/gas_mapping.launch.py').read_text():
        parser.error('Installed launch is outdated; build icir_cleanroom and source install/setup.bash first')
    manifest = prepare(root, package, args.environment, args.seed, domain, args.gui, args.empty_history)
    manifest['timeout_wall_seconds'] = args.timeout
    manifest['code_version'] = code_version()
    manifest['dry_run'] = args.dry_run
    rows = [dict(method_id=m, runner_status='not_started') for m in METHODS]
    save_summary(root, rows)
    print(f'Results: {root}', flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if args.dry_run:
        import shlex
        for run in manifest['runs']:
            print(shlex.join(run['command']))
        return 0
    exit_code = 0
    for index, run in enumerate(manifest['runs']):
        method = run['method']
        folder = root/method
        env = dict(os.environ, ROS_DOMAIN_ID=str(run['ros_domain_id']), ROS_LOCALHOST_ONLY='1',
                   GAZEBO_MASTER_URI=f'http://127.0.0.1:{free_port()}',
                   ROS_LOG_DIR=str(folder/'ros_logs'))
        run['gazebo_master_uri'] = env['GAZEBO_MASTER_URI']
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        rows[index]['runner_status'] = 'running'
        save_summary(root, rows)
        print(f'[{index+1}/7] {method} starting', flush=True)
        rows[index], stop = run_one(method, folder, run['command'], env, args.timeout)
        save_summary(root, rows)
        print(f"{method}: {rows[index]['runner_status']} / {rows[index].get('termination_reason', '')}", flush=True)
        if rows[index]['runner_status'] != 'completed':
            exit_code = 1
        if stop:
            break
    print(f'Summary: {root / "summary.csv"}', flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
