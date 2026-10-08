#!/usr/bin/env python3
"""Classify slow or failed runs into algorithm vs execution causes. Read-only."""
import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_inputs import (  # noqa: E402
    patrol_geometry, read_csv, run_folder, source_is_reachable, write_csv)

NAV_ABORT = re.compile(r'\[follow_path\] \[ActionServer\] Aborting handle')
GOAL_FAILED = re.compile(r'\[bt_navigator\].*Goal failed')
FINAL_STOP = re.compile(r'최종 추정 위치로 이동 실패\(status=(\d+)\)')
RECOVERY = re.compile(r'(Running \w*Recovery|backup|spin|clearing costmap)', re.IGNORECASE)
UNREACHABLE = re.compile(r'unreachable=(\d+)')


def final_approach_window(events):
    """Seconds between picking the final estimate and the HRS end."""
    picked = next((float(e['ros_seconds']) for e in events
                   if e['event'] == 'final_estimate_selected'), None)
    end = next((float(e['ros_seconds']) for e in events if e['event'] == 'hrs_end'), None)
    if picked is None or end is None:
        return None, picked, end
    return end - picked, picked, end


def search_navigation_losses(events):
    """Targets that drew a navigation attempt but produced no measurement."""
    attempts = sum(e['event'] == 'navigation_attempt' for e in events)
    measurements = sum(e['event'] == 'measurement_complete' for e in events)
    return attempts - measurements, attempts, measurements


def classify(row, events, console, reachable_source):
    """Separate 'the search was wrong' from 'the robot could not drive there'."""
    reason = row['termination_reason']
    error = None if row['source_position_error'] in ('', None) else float(
        row['source_position_error'])
    window, picked, end = final_approach_window(events)
    lost, attempts, measurements = search_navigation_losses(events)
    aborts = len(NAV_ABORT.findall(console))
    goal_failed = len(GOAL_FAILED.findall(console))
    final_status = FINAL_STOP.search(console)
    unreachable = UNREACHABLE.search(console)

    if 'final approach stopped' in reason:
        cause = ('execution_final_approach' if error is not None and error <= 1e-9
                 else 'mixed_final_approach_with_search_error')
        detail = ('search reached the reference cell; Nav2 could not drive the last leg'
                  if cause == 'execution_final_approach'
                  else 'Nav2 failed and the estimate was also off')
    elif reason.startswith('maximum '):
        cause, detail = 'algorithm_attempt_limit', 'search never satisfied the stop rule'
    elif lost > 0:
        cause = 'execution_search_navigation'
        detail = f'{lost} search targets produced no measurement'
    else:
        cause, detail = 'normal', 'stop rule met and the final approach succeeded'
    return dict(
        cause=cause, detail=detail,
        termination_reason=reason, source_position_error=error,
        final_status=None if final_status is None else int(final_status.group(1)),
        final_approach_seconds=window,
        final_estimate_ros=picked, hrs_end_ros=end,
        total_seconds=float(row['total_seconds']),
        search_targets=attempts, measurements=measurements,
        targets_without_measurement=lost,
        controller_aborts=aborts, bt_goal_failures=goal_failed,
        recovery_events=len(RECOVERY.findall(console)),
        controller_reported_unreachable=(None if unreachable is None
                                         else int(unreachable.group(1))),
        source_in_navigation_goal_domain=reachable_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[3]
    package = Path(__file__).resolve().parents[1]
    parser.add_argument('--raw-root', type=Path, default=workspace/'results/old')
    parser.add_argument('--summary', type=Path, default=workspace/'results/summary.csv')
    parser.add_argument('--seeds', default='',
                        help='comma separated; default = every non-normal run plus --slow')
    parser.add_argument('--slow-seconds', type=float, default=500.,
                        help='also classify runs at or above this arrival time')
    parser.add_argument('--output', type=Path,
                        default=workspace/'outputs/analysis_20261006/failure_causes')
    args = parser.parse_args()

    summary = read_csv(args.summary)
    wanted = {int(s) for s in args.seeds.split(',') if s.strip()}
    selected = [r for r in summary
                if (int(r['seed']) in wanted if wanted else
                    (r['classification'] != 'normal'
                     or float(r['total_seconds'] or 0) >= args.slow_seconds))]
    if not selected:
        parser.error('no runs selected')

    patrol = patrol_geometry(package, args.raw_root, int(selected[0]['seed']))
    cells = patrol.reachable_cells
    rows = []
    for record in sorted(selected, key=lambda r: (int(r['seed']), r['method_id'])):
        seed, method = int(record['seed']), record['method_id']
        folder = run_folder(args.raw_root, seed, method)
        if folder is None:
            rows.append(dict(seed=seed, method=method, cause='run_folder_missing'))
            continue
        run = read_csv(folder/'runs.csv')[0]
        events = read_csv(folder/'events.csv')
        console_path = args.raw_root/f'seed_{seed}'/method/'console.log'
        console = console_path.read_text(errors='ignore') if console_path.exists() else ''
        source = (float(run['source_x']), float(run['source_y']))
        distances = np.hypot(cells[:, 0]-source[0], cells[:, 1]-source[1])
        rows.append(dict(seed=seed, method=method,
                         classification=record['classification'],
                         source_x=source[0], source_y=source[1],
                         estimated_x=run['estimated_source_cell_x'],
                         estimated_y=run['estimated_source_cell_y'],
                         nearest_reachable_cell_m=float(distances.min()),
                         **classify(run, events, console,
                                    source_is_reachable(patrol, source))))

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'failure_causes.csv', rows)
    counts = collections.Counter(r['cause'] for r in rows)
    (args.output/'failure_causes.json').write_text(json.dumps(
        dict(rows=rows, counts=counts,
             inputs=dict(raw_root=str(args.raw_root), summary=str(args.summary),
                         slow_seconds=args.slow_seconds)),
        ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(dict(classified=len(rows), counts=counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
