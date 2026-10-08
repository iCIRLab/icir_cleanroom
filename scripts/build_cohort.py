#!/usr/bin/env python3
"""Evaluate every run in a seed range straight from the raw tree. Read-only inputs.

evaluate_benchmark.py froze one 45-seed cohort and its paths no longer resolve.
This rebuilds the same quantities for an arbitrary range, including the seeds that
cohort excluded, so the primary analysis can cover the whole campaign.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_inputs import (  # noqa: E402
    boundary_times, parse_seeds, patrol_geometry, read_csv, relative,
    run_folder, write_csv, workspace_root)
from icir_cleanroom.gas_mapping.application.benchmark import (  # noqa: E402
    EvaluationDomain)

METHODS = tuple(f'M{i}' for i in range(1, 8))
FAILURE_MARKER = 'final approach stopped'


def number(row, key):
    value = row.get(key)
    if value in (None, ''):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def lrs_totals(folder, run):
    """Cumulative patrol seconds for this event, from the lap records."""
    path = folder/'lrs_runs.csv'
    if not path.exists():
        return dict(cumulative_lrs_seconds=None, lrs_seconds=None)
    laps = [r for r in read_csv(path) if r['event_id'] == run['event_id']]
    spans = []
    for lap in laps:
        start, end = number(lap, 'start_ros_seconds'), number(lap, 'end_ros_seconds')
        if start is not None and end is not None and end >= start:
            spans.append(end-start)
    return dict(cumulative_lrs_seconds=math.fsum(spans) if spans else None,
                lrs_seconds=spans[-1] if spans else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = workspace_root()
    parser.add_argument('--raw-root', type=Path, default=Path('results/old'),
                        help='relative to the workspace root unless absolute')
    parser.add_argument('--seeds', default='all')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--output', type=Path, default=Path('outputs/analysis_20261006'))
    args = parser.parse_args()
    raw_root = args.raw_root if args.raw_root.is_absolute() else root/args.raw_root
    output = args.output if args.output.is_absolute() else root/args.output
    methods = tuple(m.strip() for m in args.methods.split(',') if m.strip())

    available = sorted(int(p.name.split('_')[1]) for p in raw_root.glob('seed_*')
                       if p.name.split('_')[1].isdigit())
    seeds = parse_seeds(args.seeds, available)
    if not seeds:
        parser.error(f'no seed folders matched under {raw_root}')

    patrol = patrol_geometry(Path(__file__).resolve().parents[1], raw_root, seeds[0])
    domain = EvaluationDomain.snapshot(patrol.goal_geometry, patrol.goal_mask,
                                       patrol.field_geometry)
    rows, missing = [], []
    for seed in seeds:
        for method in methods:
            folder = run_folder(raw_root, seed, method)
            if folder is None:
                missing.append(dict(seed=seed, method=method, reason='no runs.csv'))
                continue
            records = read_csv(folder/'runs.csv')
            if len(records) != 1:
                missing.append(dict(seed=seed, method=method,
                                    reason=f'{len(records)} rows in runs.csv'))
                continue
            run = records[0]
            events = read_csv(folder/'events.csv')
            source = (number(run, 'source_x'), number(run, 'source_y'))
            estimate = (number(run, 'estimated_source_cell_x'),
                        number(run, 'estimated_source_cell_y'))
            reason = run['termination_reason']
            reference = domain.reference(source) if None not in source else None
            evaluation = (reference.evaluate(
                None if None in estimate else estimate,
                None if None in source else source) if reference is not None else {})
            times = boundary_times(events)
            entry = dict(
                seed=seed, method=method, run_id=run['run_id'],
                schema_version=run['schema_version'],
                termination_reason=reason,
                navigation_failed=FAILURE_MARKER in reason,
                attempt_limit=reason.startswith('maximum '),
                T=number(run, 'total_seconds'),
                initial_seconds=number(run, 'initial_seconds'),
                search_seconds=number(run, 'search_seconds'),
                E_raw=number(run, 'source_position_error'),
                D=number(run, 'hrs_distance_m'),
                source_x=source[0], source_y=source[1],
                estimated_x=estimate[0], estimated_y=estimate[1],
                attempt_count=number(run, 'attempt_count'),
                search_iterations=number(run, 'search_iterations'),
                measurement_count=number(run, 'measurement_count'),
                best_measured_concentration=number(run, 'best_measured_concentration'),
                hrs_start_ros=times['hrs_start'], hrs_end_ros=times['hrs_end'],
                first_measurement_ros=times['first_measurement'],
                final_estimate_ros=times['final_estimate'],
                final_approach_end_ros=times['final_approach_end'],
                final_approach_succeeded=times['final_approach_succeeded'],
                final_approach_reason=times['final_approach_reason'],
                runs_csv=relative(folder/'runs.csv', root),
                **{k: v for k, v in evaluation.items()},
                **lrs_totals(folder, run))
            entry['E'] = entry.get('adjusted_source_position_error')
            entry['source_unreachable'] = bool(entry.get('evaluation_reference_adjusted'))
            # Primary metric: the search itself, ending when the estimate is
            # fixed. The drive to that estimate is a separate measurement.
            entry['T_decision'] = (
                None if None in (times['hrs_start'], times['final_estimate'])
                else times['final_estimate'] - times['hrs_start'])
            entry['T_approach'] = (
                None if None in (times['final_estimate'], times['hrs_end'])
                else times['hrs_end'] - times['final_estimate'])
            succeeded = times['final_approach_succeeded']
            if succeeded is None:  # logs written before the event existed
                succeeded = not entry['navigation_failed']
            entry['approach_succeeded'] = bool(succeeded)
            rows.append(entry)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'runs_evaluated.csv', rows)
    report = dict(schema_version=1, raw_root=relative(raw_root, root),
                  seeds=seeds, methods=list(methods), run_count=len(rows),
                  missing=missing,
                  navigation_failures=sum(r['navigation_failed'] for r in rows),
                  attempt_limits=sum(r['attempt_limit'] for r in rows),
                  unreachable_sources=sorted({r['seed'] for r in rows
                                              if r['source_unreachable']}),
                  patrol_points=patrol.patrol_points)
    (output/'runs_evaluated.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'patrol_points'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
