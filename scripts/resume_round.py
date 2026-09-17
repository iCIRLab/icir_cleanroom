#!/usr/bin/env python3
"""Resume one round of run_methods_random.py from a given method onward.

Reuses the round's already-frozen manifest.json/run_config.yaml (so the
shared random gas source and every method's command are exactly what
prepare_random() generated originally); earlier methods keep their already
recorded result untouched. Use this after a method got stuck (e.g. a Nav2
lifecycle bringup flake) so the rest of the round doesn't need a fresh
random source. This program launches ROS; importing it does not.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_methods_once import FIELDS, METHODS, free_port, read_result, save_summary  # noqa: E402
from run_methods_random import run_one_paused  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round_dir', type=Path, help='Round directory containing manifest.json')
    parser.add_argument('--from-method', choices=METHODS, default=None,
                         help='First method to (re)run; earlier methods keep their existing result (default: first not completed)')
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args(argv)

    root = args.round_dir.expanduser().resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    runs = manifest['runs']

    if args.from_method is not None:
        start_index = METHODS.index(args.from_method)
    else:
        start_index = next(
            (i for i, run in enumerate(runs)
             if read_result(root/run['method']/'hrs', run['method'])[0] is None),
            len(runs))

    rows = []
    for index, run in enumerate(runs):
        method = run['method']
        if index < start_index:
            row, _ = read_result(root/method/'hrs', method)
            rows.append({key: row.get(key, '') for key in FIELDS} if row
                        else dict(method_id=method, runner_status='completed'))
    save_summary(root, rows + [dict(method_id=run['method'], runner_status='not_started')
                               for run in runs[start_index:]])

    exit_code = 0
    for index in range(start_index, len(runs)):
        run = runs[index]
        method = run['method']
        folder = root/method
        env = dict(os.environ, ROS_DOMAIN_ID=str(run['ros_domain_id']), ROS_LOCALHOST_ONLY='1',
                   GAZEBO_MASTER_URI=f'http://127.0.0.1:{free_port()}',
                   ROS_LOG_DIR=str(folder/'ros_logs'))
        print(f'[{index+1}/{len(runs)}] {method} starting', flush=True)
        last = index == len(runs)-1
        row, stop = run_one_paused(method, folder, run['command'], env, args.timeout,
                                   args.gui and not last)
        rows.append(row)
        save_summary(root, rows + [dict(method_id=runs[i]['method'], runner_status='not_started')
                                   for i in range(index+1, len(runs))])
        print(f"{method}: {row['runner_status']} / {row.get('outcome', 'no HRS result')} / "
              f"{row.get('termination_reason', '')}", flush=True)
        if row['runner_status'] != 'completed':
            exit_code = 1
        if stop:
            print(f'{method} was interrupted; stopping.', flush=True)
            break
    print(f'Summary: {root/"summary.csv"}', flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
