#!/usr/bin/env python3
"""Run M1..M7 sequentially for a range of consecutive seeds
(start_seed, start_seed+1, ..., start_seed+count-1), one round per seed.

Same idea as run_methods_random.py --rounds, but with deterministic
sequential seeds instead of a fresh random seed drawn each round, and using
the same results/seed_<seed>/ folder layout as run_methods_seeded.py (so a
seed already run one-method-at-a-time with that script is picked up here
too: methods that already have a completed result are skipped and kept).
Within one seed, all 7 methods share that seed's gas source, exactly like a
run_methods_random.py round.

With --gui, after each method finishes the batch pauses and waits for you to
press Enter before launching the next run, so you can take your own
screenshot of the still-open RViz window; the very last method of the very
last seed is left running instead of paused, since there is nothing after it
to hand off to. Run this directly in your own terminal (not backgrounded),
since the Enter prompts need a real interactive stdin. This program launches
ROS; importing it does not.
"""
import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_methods_once import METHODS, free_port, read_result, save_summary  # noqa: E402
from run_methods_random import RANDOM_SOURCE_MODES, prepare_random, run_one_paused  # noqa: E402


def run_seed_round(root, package, environment, seed, source_mode, domain, gui,
                   empty_history, timeout, is_last_seed, pause=True):
    """Run (or resume) M1..M7 for one seed; return (exit_code, stopped)."""
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['seed'] != seed or manifest['source_mode'] != source_mode:
            raise ValueError(f'{root} already holds a different seed/source_mode')
    else:
        manifest = prepare_random(root, package, environment, seed, source_mode, domain, gui, empty_history)
    rows = []
    for run in manifest['runs']:
        existing, _ = read_result(root/run['method']/'hrs', run['method'])
        if existing:
            row = dict(existing)
            row['runner_status'] = 'completed'
        else:
            row = dict(method_id=run['method'], runner_status='not_started')
        rows.append(row)
    save_summary(root, rows)
    print(f'Results: {root}', flush=True)
    print(f'Gas source this seed: source_mode={source_mode}, seed={seed} (shared by M1..M7)', flush=True)

    exit_code = 0
    stopped = False
    for index, run in enumerate(manifest['runs']):
        method = run['method']
        if rows[index].get('runner_status') == 'completed':
            print(f'[{index+1}/7] {method} already completed; skipping', flush=True)
            continue
        folder = root/method
        env = dict(os.environ, ROS_DOMAIN_ID=str(run['ros_domain_id']), ROS_LOCALHOST_ONLY='1',
                  GAZEBO_MASTER_URI=f'http://127.0.0.1:{free_port()}', ROS_LOG_DIR=str(folder/'ros_logs'))
        print(f'[{index+1}/7] {method} starting', flush=True)
        last_of_batch = is_last_seed and index == len(manifest['runs'])-1
        rows[index], stop = run_one_paused(method, folder, run['command'], env, timeout,
                                           gui and pause and not last_of_batch)
        save_summary(root, rows)
        print(f"{method}: {rows[index]['runner_status']} / {rows[index].get('outcome', 'no HRS result')} / "
              f"{rows[index].get('termination_reason', '')}", flush=True)
        if rows[index]['runner_status'] != 'completed':
            exit_code = 1
        if stop:
            stopped = True
            break
    print(f'Summary: {root/"summary.csv"}', flush=True)
    return exit_code, stopped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--environment', default='aws_small_warehouse')
    parser.add_argument('--start-seed', type=int, required=True,
                         help='First seed to run; M1..M7 inside it share that seed\'s source')
    parser.add_argument('--count', type=int, default=1,
                         help='Number of consecutive seeds to run: start_seed, start_seed+1, ... (default 1)')
    parser.add_argument('--source-mode', choices=RANDOM_SOURCE_MODES, default='random_after_peak')
    parser.add_argument('--output-root', type=Path, default=None,
                         help='Parent directory for the seed_<seed> folders (default: ./results)')
    parser.add_argument('--ros-domain-id', type=int, default=None,
                         help='First of seven private ROS domains (0..94); reused for every seed in the '
                              'range (safe since seeds run strictly one after another). Default: a fresh '
                              'random domain per seed')
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--no-pause', action='store_true',
                         help='With --gui, skip the Enter-to-continue prompt and move straight to the '
                              'next method/seed instead of pausing for a screenshot')
    parser.add_argument('--empty-history', action='store_true')
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error('count must be at least 1')
    if args.start_seed < 0:
        parser.error('start-seed must be nonnegative')
    if args.ros_domain_id is not None and not 0 <= args.ros_domain_id <= 94:
        parser.error('ros-domain-id must be in 0..94')

    from ament_index_python.packages import get_package_share_directory
    package = Path(get_package_share_directory('icir_cleanroom'))
    output_root = (args.output_root or Path.cwd()/'results').expanduser().resolve()

    exit_code = 0
    for offset in range(args.count):
        seed = args.start_seed + offset
        domain = args.ros_domain_id if args.ros_domain_id is not None else random.SystemRandom().randrange(20, 90)
        root = output_root/f'seed_{seed}'
        print(f'== Seed {seed} ({offset+1}/{args.count}) ==', flush=True)
        round_exit_code, stopped = run_seed_round(
            root, package, args.environment, seed, args.source_mode, domain,
            args.gui, args.empty_history, args.timeout, offset == args.count-1,
            pause=not args.no_pause)
        exit_code = exit_code or round_exit_code
        if stopped:
            print(f'Seed {seed} was interrupted; stopping.', flush=True)
            break
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
