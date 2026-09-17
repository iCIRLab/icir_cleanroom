#!/usr/bin/env python3
"""Run one method at a time by hand, sharing one gas source across M1..M7.

Unlike run_methods_random.py (which loops M1..M7 automatically in one
process), this is meant to be invoked once per method, directly in your own
terminal: run it for M1, look at RViz/take your screenshot, Ctrl+C or press
Enter when the prompt asks, then run it again for M2 with the same --seed
(and --output if you don't want the default seed-keyed folder), and so on
through M7. Because the same (environment, seed, source_mode) always
regenerates the same random gas source, every method you run this way shares
one source, exactly like one run_methods_random.py round would.

The round's manifest/run_config is created on the first method you run for a
given seed and reused (not recreated) for the rest; --gui/--empty-history
only take effect on that first call. This program launches ROS; importing it
does not.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_methods_once import FIELDS, METHODS, free_port, read_result, save_summary  # noqa: E402
from run_methods_random import RANDOM_SOURCE_MODES, prepare_random, run_one_paused  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--environment', default='aws_small_warehouse')
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--seed', type=int, required=True,
                         help='Shared source RNG seed; reuse the same value for every method in this round')
    parser.add_argument('--source-mode', choices=RANDOM_SOURCE_MODES, default='random_after_peak')
    parser.add_argument('--output', type=Path, default=None,
                         help='Round directory; default results/seed_<seed> so repeated calls with the '
                              'same --seed reuse it automatically')
    parser.add_argument('--ros-domain-id', type=int, default=None, help='First of seven private ROS domains (0..94); only used the first time this round is created')
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--gui', action='store_true', help='Only takes effect the first time this round is created')
    parser.add_argument('--empty-history', action='store_true', help='Only takes effect the first time this round is created')
    args = parser.parse_args(argv)

    from ament_index_python.packages import get_package_share_directory
    package = Path(get_package_share_directory('icir_cleanroom'))

    root = (args.output or Path.cwd()/'results'/f'seed_{args.seed}').expanduser().resolve()
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['seed'] != args.seed or manifest['source_mode'] != args.source_mode:
            parser.error(f'{root} already holds a different seed/source_mode; pass --output for a new folder')
    else:
        domain = args.ros_domain_id if args.ros_domain_id is not None else __import__('random').SystemRandom().randrange(20, 90)
        manifest = prepare_random(root, package, args.environment, args.seed,
                                  args.source_mode, domain, args.gui, args.empty_history)
        print(f'New round created: {root} (seed={args.seed}, source_mode={args.source_mode})', flush=True)

    run = next(r for r in manifest['runs'] if r['method'] == args.method)
    folder = root/args.method
    env = dict(os.environ, ROS_DOMAIN_ID=str(run['ros_domain_id']), ROS_LOCALHOST_ONLY='1',
              GAZEBO_MASTER_URI=f'http://127.0.0.1:{free_port()}',
              ROS_LOG_DIR=str(folder/'ros_logs'))
    print(f'{args.method} starting (seed={args.seed}, source_mode={args.source_mode})', flush=True)
    row, _ = run_one_paused(args.method, folder, run['command'], env, args.timeout, pause=True)
    print(f"{args.method}: {row['runner_status']} / {row.get('outcome', 'no HRS result')} / "
          f"{row.get('termination_reason', '')}", flush=True)

    rows = []
    for candidate in METHODS:
        if candidate == args.method:
            rows.append(row)
            continue
        existing, _ = read_result(root/candidate/'hrs', candidate)
        if existing:
            existing_row = {key: existing.get(key, '') for key in FIELDS}
            existing_row['runner_status'] = 'completed'
        else:
            existing_row = dict(method_id=candidate, runner_status='not_started')
        rows.append(existing_row)
    save_summary(root, rows)
    print(f'Summary: {root/"summary.csv"}', flush=True)
    return 0 if row['runner_status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
