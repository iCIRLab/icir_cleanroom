#!/usr/bin/env python3
"""Run M1..M7 once sequentially with a random gas source shared by all methods
this round. This program launches ROS; importing it does not.

Wraps the same launch/tracking/summary machinery as run_methods_once.py, but
switches the environment profile's source_mode to a random generator and
draws the RNG seed fresh (via random.SystemRandom) on every invocation unless
--seed is given. Because M1..M7 in one invocation all deep-copy the same
profile, an identical source_mode/seed means an identical randomly generated
gas source across the seven methods; the next invocation draws a new seed, so
the source differs round to round.

With --gui, after each method finishes the batch pauses and waits for you to
press Enter before launching the next run, so you can take your own
screenshot of the still-open RViz window (skipped when headless, since there
is nothing on screen to look at).
"""
import argparse
import copy
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_methods_once import (  # noqa: E402
    FIELDS, METHODS, OwnedLaunch, free_port, read_result, save_summary)

RANDOM_SOURCE_MODES = ('random_after_peak', 'recurrent_hotspots_after_peak')


def run_one_paused(method, folder, command, env, timeout, pause, poll=.5):
    """Same as run_methods_once.run_one, but (when `pause`) waits for Enter
    right after a run ends and before Gazebo/RViz are torn down, so the
    still-open window can be screenshotted by hand."""
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
            if pause:
                input(f'{method} finished ({result["runner_status"]}); take your '
                      'screenshot, then press Enter to continue '
                      '(Ctrl+C to stop the batch)... ')
        except (KeyboardInterrupt, EOFError):
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
        elif result['runner_status'] == 'completed':
            result.update(runner_status='runner_error', note='Completed record disappeared')
    except (OSError, ValueError) as error:
        if result['runner_status'] != 'cleanup_failed':
            result.update(runner_status='runner_error', note=str(error))
    return result, interrupted or result['runner_status'] == 'cleanup_failed'


def prepare_random(root, package, environment, seed, source_mode, domain, gui, empty_history):
    """Same freeze as run_methods_once.prepare, plus a shared random source override."""
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
    # One random source per round: every method below deep-copies this same
    # profile, so an identical source_mode/seed here produces an identical
    # randomly generated gas source for M1..M7; no persisted-source override.
    profile['gas_environment'].update(
        source_mode=source_mode, source_random_seed=seed, persist_source_state=False)
    controller.update(save_history=False, repeat_after_hrs=False)
    for name, content in [('environment.yaml', profile), ('mapping.yaml', controller)]:
        (snapshot/name).write_text(yaml.safe_dump(content, sort_keys=False))
    # Preserve human-readable navigation settings as well as the exact source paths.
    shutil.copy2(package/'config/nav2_params.yaml', snapshot/'nav2_params.yaml')
    shutil.copy2(package/profile['environment']['navigation_profile'], snapshot/'navigation_profile.yaml')
    profile['environment']['navigation_profile'] = str(snapshot/'navigation_profile.yaml')
    manifest = dict(environment=environment, methods=METHODS, seed=seed, source_mode=source_mode,
                    ros_domain_base=domain, gui=gui, package=str(package),
                    initial_history=str(original_history), history_copied=copied,
                    save_history=False, repeat_after_hrs=False,
                    note=(f'Random gas source (source_mode={source_mode}, seed={seed}) shared by '
                          'all methods this round; live navigation and LRS measurements are not '
                          'deterministic replays.'),
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


def run_round(root, package, environment, seed, source_mode, domain, gui,
              empty_history, timeout, dry_run, is_last_round):
    """Run M1..M7 once each under one shared random source; return (exit_code, stopped)."""
    manifest = prepare_random(root, package, environment, seed, source_mode, domain, gui, empty_history)
    manifest['timeout_wall_seconds'] = timeout
    manifest['dry_run'] = dry_run
    rows = [dict(method_id=m, runner_status='not_started') for m in METHODS]
    save_summary(root, rows)
    print(f'Results: {root}', flush=True)
    print(f'Gas source this round: source_mode={source_mode}, seed={seed} (shared by M1..M7)', flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if dry_run:
        import shlex
        for run in manifest['runs']:
            print(shlex.join(run['command']))
        return 0, False
    exit_code = 0
    stopped = False
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
        last_of_batch = is_last_round and index == len(manifest['runs'])-1
        rows[index], stop = run_one_paused(
            method, folder, run['command'], env, timeout, gui and not last_of_batch)
        save_summary(root, rows)
        print(f"{method}: {rows[index]['runner_status']} / {rows[index].get('outcome', 'no HRS result')} / {rows[index].get('termination_reason', '')}", flush=True)
        if rows[index]['runner_status'] != 'completed':
            exit_code = 1
        if stop:
            stopped = True
            break
    print(f'Summary: {root / "summary.csv"}', flush=True)
    return exit_code, stopped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', default='aws_small_warehouse')
    parser.add_argument('--output', type=Path,
                         help='New output directory (batch directory when --rounds > 1); existing directory is rejected')
    parser.add_argument('--timeout', type=float, default=1800., help='Maximum wall seconds per method, including startup and LRS (default 1800)')
    parser.add_argument('--rounds', type=int, default=1,
                         help='Number of rounds to repeat automatically; each round draws its own random '
                              'seed/source and all 7 methods inside it share that source (default 1)')
    parser.add_argument('--source-mode', choices=RANDOM_SOURCE_MODES, default='random_after_peak',
                         help='Random source generator shared by all 7 methods this round (default random_after_peak)')
    parser.add_argument('--seed', type=int, default=None,
                         help='Common source RNG seed for this round (same source for M1..M7); '
                              'default draws a fresh random seed every round. Only valid with --rounds 1, '
                              'since one seed would repeat the same source every round otherwise')
    parser.add_argument('--ros-domain-id', type=int, default=None, help='First of seven private ROS domains (0..94)')
    parser.add_argument('--gui', action='store_true', help='Show Gazebo and RViz; default headless')
    parser.add_argument('--empty-history', action='store_true', help='Start all methods without existing history; original file stays untouched')
    parser.add_argument('--dry-run', action='store_true', help='Prepare configs and summaries, print commands; start no processes')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive and finite')
    if args.rounds < 1:
        parser.error('rounds must be at least 1')
    if args.seed is not None:
        if args.seed < 0:
            parser.error('seed must be nonnegative')
        if args.rounds > 1:
            parser.error('--seed fixes one source; omit it (or use --rounds 1) so each round gets its own')
    if args.ros_domain_id is not None and not 0 <= args.ros_domain_id <= 94:
        parser.error('ros-domain-id must be in 0..94')
    from ament_index_python.packages import get_package_share_directory
    package = Path(get_package_share_directory('icir_cleanroom'))
    if "LaunchConfiguration('run_config')" not in (package/'launch/gas_mapping.launch.py').read_text():
        parser.error('Installed launch is outdated; build icir_cleanroom and source install/setup.bash first')
    batch_root = (args.output or Path.cwd()/'results'/datetime.now(timezone.utc).strftime('methods_random_%Y%m%dT%H%M%S_%fZ')).expanduser().resolve()
    single_round = args.rounds == 1
    if not single_round:
        batch_root.mkdir(parents=True, exist_ok=False)
    exit_code = 0
    batch_manifest = dict(rounds=[])
    for round_index in range(args.rounds):
        domain = args.ros_domain_id if args.ros_domain_id is not None else random.SystemRandom().randrange(20, 90)
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2_000_000_000)
        root = batch_root if single_round else batch_root/f'round_{round_index+1:02d}_seed{seed}'
        if not single_round:
            print(f'== Round {round_index+1}/{args.rounds} ==', flush=True)
        round_exit_code, stopped = run_round(
            root, package, args.environment, seed, args.source_mode, domain,
            args.gui, args.empty_history, args.timeout, args.dry_run,
            round_index == args.rounds-1)
        exit_code = exit_code or round_exit_code
        if not single_round:
            batch_manifest['rounds'].append(dict(
                round=round_index+1, seed=seed, source_mode=args.source_mode,
                directory=str(root), exit_code=round_exit_code, stopped=stopped))
            (batch_root/'batch_manifest.json').write_text(json.dumps(batch_manifest, indent=2)+'\n')
        if stopped:
            print(f'Round {round_index+1} was interrupted; stopping the batch.', flush=True)
            break
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
