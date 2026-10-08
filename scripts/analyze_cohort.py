#!/usr/bin/env python3
"""Aggregate a finished seed campaign over any seed range. Inputs stay read-only."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_inputs import (  # noqa: E402
    best_measurement_milestone, boundary_times, first_target_cell,
    nearest_patrol_distance, parse_seeds, patrol_geometry,
    patrol_peak_concentration, read_csv, recover_source_truth, relative,
    run_folder, workspace_root, write_csv)
from cohort_plots import (  # noqa: E402
    decision_time_cdf, error_cdf, stage_decomposition, time_error_panels)
from cohort_stats import (  # noqa: E402
    VERDICT_RULE, bootstrap_interval, describe, paired_comparisons,
    paired_differences, rank_stability, threshold_table, weighted_scores)

# 2x2 design: start position x search strategy. M3/M4 are folded into M1/M2 because
# their first target cell is always within one cell of M1/M2 (see first_cell_agreement).
DESIGN = {
    'M1': dict(start='patrol_peak', search='estimate_max'),
    'M2': dict(start='patrol_peak', search='spiral'),
    'M5': dict(start='patrol_centroid', search='estimate_max'),
    'M6': dict(start='patrol_centroid', search='spiral'),
}
DEFAULT_METHODS = ('M1', 'M2', 'M5', 'M6')
MERGED_METHODS = {'M3': 'M1', 'M4': 'M2'}
ALL_METHODS = tuple(f'M{i}' for i in range(1, 8))
REFERENCE_METHOD = 'M1'
# Decision time (search transition -> final estimate fixed) is the primary
# time metric: it ends where the search ends. Total time and the final
# approach are reported next to it, not folded into it.
PRIMARY_METRICS = ('T_decision', 'E')
SECONDARY_METRICS = ('T', 'T_approach')
# Nothing is censored on decision time: a run that later failed to drive to
# its estimate still produced that estimate at a measured time. Total time
# keeps the censoring, because a failed approach ends on the retry budget.
CENSORED_METRICS = ('T',)
# The 2x2 design's own contrasts: search strategy at each start, then start
# position at each search strategy.
DESIGN_PAIRS = (('M5', 'M1'), ('M6', 'M2'), ('M2', 'M1'), ('M6', 'M5'))
# Error thresholds a reader may require; no single one is the success criterion.
ERROR_THRESHOLDS = (0., .25, .5, 1., 2., 5.)
# Same-first-cell pairs: the merge evidence, then the design's own contrasts.
START_PAIRS = (('M1', 'M3'), ('M2', 'M4'), ('M1', 'M2'), ('M3', 'M4'),
               ('M5', 'M6'), ('M1', 'M5'), ('M2', 'M6'))


def numeric(row, key):
    value = row.get(key)
    if value in (None, ''):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_runs(path, methods, seeds_text):
    """Read the rebuilt evaluation table and mark navigation-failed arrival times."""
    raw = read_csv(path)
    available = {int(r['seed']) for r in raw}
    seeds = parse_seeds(seeds_text, available)
    rows = []
    for r in raw:
        seed, method = int(r['seed']), r['method']
        if seed not in seeds or method not in methods:
            continue
        failed = str(r.get('navigation_failed', '')).lower() == 'true'
        rows.append(dict(
            seed=seed, method=method, run_id=r['run_id'],
            termination_reason=r.get('termination_reason', ''),
            navigation_failed=failed,
            attempt_limit=str(r.get('attempt_limit', '')).lower() == 'true',
            # The arrival time of a failed approach measures the retry budget.
            censored=failed,
            T=numeric(r, 'T'), E=numeric(r, 'E'), E_raw=numeric(r, 'E_raw'),
            T_decision=numeric(r, 'T_decision'),
            T_approach=numeric(r, 'T_approach'),
            approach_succeeded=str(
                r.get('approach_succeeded', '')).lower() == 'true',
            D=numeric(r, 'D'),
            source_x=numeric(r, 'source_x'), source_y=numeric(r, 'source_y'),
            estimated_x=numeric(r, 'estimated_x'), estimated_y=numeric(r, 'estimated_y'),
            source_unreachable=str(r.get('source_unreachable', '')).lower() == 'true',
            reference_distance_m=numeric(r, 'evaluation_reference_distance_m'),
            lrs_seconds=numeric(r, 'cumulative_lrs_seconds'),
            hrs_start_ros=numeric(r, 'hrs_start_ros'),
            hrs_end_ros=numeric(r, 'hrs_end_ros'),
            first_measurement_ros=numeric(r, 'first_measurement_ros'),
            final_estimate_ros=numeric(r, 'final_estimate_ros'),
            best_measured_concentration=numeric(r, 'best_measured_concentration')))
    return rows, seeds


def method_statistics(rows, methods):
    """Distributions plus the exact-hit split, because E is close to bimodal."""
    table = []
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        entry = dict(method=method, n=len(subset))
        for metric in ('T_decision', 'T_approach', 'T', 'E', 'E_raw', 'D'):
            entry.update(describe([r[metric] for r in subset], metric))
        completed = [r for r in subset if not r['censored']]
        entry['navigation_failed'] = sum(r['navigation_failed'] for r in subset)
        entry['navigation_failure_rate'] = (
            entry['navigation_failed'] / len(subset) if subset else None)
        entry.update(describe([r['T'] for r in completed], 'T_completed'))
        arrived = [r for r in subset if r['approach_succeeded']]
        entry['approach_succeeded_count'] = len(arrived)
        entry['approach_success_rate'] = (
            len(arrived)/len(subset) if subset else None)
        entry.update(describe(
            [r['T_approach'] for r in arrived], 'T_approach_arrived'))
        errors = [r['E'] for r in subset if r['E'] is not None]
        entry.update({'E_' + k: v for k, v in
                      threshold_table(errors, ERROR_THRESHOLDS).items()})
        entry.update({'E_median_' + k: v for k, v in
                      bootstrap_interval(errors).items()})
        entry.update({'T_median_' + k: v for k, v in
                      bootstrap_interval([r['T'] for r in subset]).items()})
        entry.update({'T_decision_median_' + k: v for k, v in
                      bootstrap_interval(
                          [r['T_decision'] for r in subset]).items()})
        exact = [r for r in subset if r['E'] is not None and r['E'] <= 1e-9]
        missed = [r for r in subset if r['E'] is not None and r['E'] > 1e-9]
        entry['E_exact_hits'] = len(exact)
        entry['E_missed'] = len(missed)
        entry.update(describe([r['T'] for r in exact], 'T_given_exact'))
        entry.update(describe([r['T'] for r in missed], 'T_given_missed'))
        entry.update(describe([r['E'] for r in missed], 'E_given_missed'))
        table.append(entry)
    return table


def stage_metrics(rows, raw_root, patrol):
    """First-cell distance, time to the best measurement, and the tail after it."""
    out = []
    for row in rows:
        folder = run_folder(raw_root, row['seed'], row['method'])
        entry = dict(seed=row['seed'], method=row['method'], run_id=row['run_id'],
                     navigation_failed=row['navigation_failed'],
                     first_target_x=None, first_target_y=None,
                     first_target_source_distance_m=None,
                     first_target_patrol_distance_m=None,
                     time_to_first_measurement=None,
                     time_first_measurement_to_best=None,
                     time_best_to_end=None, best_concentration=None,
                     stage_status='events_missing')
        if folder is None:
            out.append(entry)
            continue
        events = read_csv(folder/'events.csv')
        bounds = boundary_times(events)
        cell, _ = first_target_cell(events)
        source = (row['source_x'], row['source_y'])
        if cell is not None:
            entry.update(first_target_x=cell[0], first_target_y=cell[1])
            if None not in source:
                entry['first_target_source_distance_m'] = math.dist(cell, source)
            entry['first_target_patrol_distance_m'] = nearest_patrol_distance(
                patrol.patrol_points, cell)
        history_path = folder/'estimate_history.csv'
        history = read_csv(history_path) if history_path.exists() else []
        best_time, best_value = best_measurement_milestone(history)
        start, first, end = (bounds['hrs_start'], bounds['first_measurement'],
                             bounds['hrs_end'])
        entry['best_concentration'] = best_value
        if None not in (start, first):
            entry['time_to_first_measurement'] = first - start
        if None not in (first, best_time):
            entry['time_first_measurement_to_best'] = best_time - first
        if None not in (best_time, end):
            # A failed final approach inflates this tail with the retry budget.
            entry['time_best_to_end'] = end - best_time
        entry['stage_status'] = 'complete' if None not in (
            start, first, end, best_time) else 'partial'
        out.append(entry)
    return out


def stage_summary(stage_rows, methods):
    fields = ('first_target_source_distance_m', 'time_to_first_measurement',
              'time_first_measurement_to_best')
    table = []
    for method in methods:
        subset = [r for r in stage_rows if r['method'] == method]
        arrived = [r for r in subset if not r['navigation_failed']]
        entry = dict(method=method, n=len(subset), arrived=len(arrived),
                     complete=sum(r['stage_status'] == 'complete' for r in subset))
        for field in fields:
            entry.update(describe([r[field] for r in subset], field))
        # Excludes failed approaches, whose tail records the retry budget.
        entry.update(describe([r['time_best_to_end'] for r in arrived], 'time_best_to_end'))
        table.append(entry)
    return table


# Outcomes the campaign can classify; zero counts must still be reported.
COMPLETION_LABELS = ('normal', 'final_approach_failed', 'attempt_limit',
                     'timeout', 'interrupted', 'missing')


def completion_rates(rows, methods, all_methods_rows=None):
    """Navigation outcomes per method, counted over every run in the range."""
    table = []
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        failed = [r for r in subset if r['navigation_failed']]
        table.append(dict(
            method=method, runs=len(subset),
            normal=len(subset)-len(failed)-sum(r['attempt_limit'] for r in subset),
            final_approach_failed=len(failed),
            attempt_limit=sum(r['attempt_limit'] for r in subset),
            final_approach_failed_rate=len(failed)/len(subset) if subset else None,
            failed_seeds=sorted(r['seed'] for r in failed),
            # A failed approach still reports where the search had arrived.
            failed_with_zero_error=sum(
                1 for r in failed if r['E'] is not None and r['E'] <= 1e-9)))
    return table


def first_cell_agreement(stage_rows, resolution, pairs=START_PAIRS):
    """How often two methods opened on the same cell, or one cell apart."""
    by_key = {(r['seed'], r['method']): r for r in stage_rows}
    seeds = sorted({r['seed'] for r in stage_rows})
    table = []
    for left, right in pairs:
        same = within_one = compared = 0
        distances = []
        for seed in seeds:
            a, b = by_key.get((seed, left)), by_key.get((seed, right))
            if a is None or b is None:
                continue
            if a['first_target_x'] is None or b['first_target_x'] is None:
                continue
            compared += 1
            distance = math.dist((a['first_target_x'], a['first_target_y']),
                                 (b['first_target_x'], b['first_target_y']))
            distances.append(distance)
            if distance <= 1e-9:
                same += 1
            # One cell apart includes the diagonal neighbour.
            if distance <= resolution * math.sqrt(2) + 1e-9:
                within_one += 1
        table.append(dict(
            method_a=left, method_b=right, compared=compared,
            identical=same, identical_rate=same / compared if compared else None,
            within_one_cell=within_one,
            within_one_cell_rate=within_one / compared if compared else None,
            **describe(distances, 'first_cell_distance')))
    return table


def seed_conditions(rows, patrol, raw_root, methods, resolution):
    """Per-seed difficulty factors, independent of which method ran."""
    by_seed = collections.defaultdict(list)
    for row in rows:
        by_seed[row['seed']].append(row)
    cells = patrol.reachable_cells
    table = []
    for seed in sorted(by_seed):
        sample = by_seed[seed][0]
        source = (sample['source_x'], sample['source_y'])
        truth = recover_source_truth(raw_root, seed, methods)
        adjusted = any(r['source_unreachable'] for r in by_seed[seed])
        distances = np.hypot(cells[:, 0] - source[0], cells[:, 1] - source[1])
        entry = dict(
            seed=seed, source_x=source[0], source_y=source[1],
            nearest_patrol_distance_m=nearest_patrol_distance(patrol.patrol_points, source),
            source_unreachable=adjusted,
            nearest_reachable_cell_distance_m=float(distances.min()),
            reference_distance_m=sample['reference_distance_m'], **truth)
        entry['patrol_peak_concentration'] = patrol_peak_concentration(
            patrol.patrol_points, source, truth.get('source_sigma'),
            truth.get('source_strength'))
        if truth.get('source_log_x') is not None:
            entry['source_log_matches_runs_csv'] = (
                math.isclose(truth['source_log_x'], source[0], abs_tol=1e-6)
                and math.isclose(truth['source_log_y'], source[1], abs_tol=1e-6))
        table.append(entry)
    return table


def condition_bins(rows, conditions, methods, metrics=PRIMARY_METRICS):
    """Split methods by each condition so results read per situation."""
    by_seed = {c['seed']: c for c in conditions}
    groups = []

    def add(name, label, predicate):
        subset = [r for r in rows if predicate(by_seed.get(r['seed'], {}))]
        for method in methods:
            values = [r for r in subset if r['method'] == method]
            entry = dict(condition=name, bin=label, method=method, n=len(values))
            for metric in metrics:
                entry.update(describe([v[metric] for v in values], metric))
            groups.append(entry)

    add('source_reachability', 'reachable', lambda c: c.get('source_unreachable') is False)
    add('source_reachability', 'unreachable', lambda c: c.get('source_unreachable') is True)
    patrol_values = [c['nearest_patrol_distance_m'] for c in conditions]
    cut = statistics.median(patrol_values)
    add('nearest_patrol_distance', f'<= median ({cut:.3f} m)',
        lambda c: c.get('nearest_patrol_distance_m') is not None
        and c['nearest_patrol_distance_m'] <= cut)
    add('nearest_patrol_distance', f'> median ({cut:.3f} m)',
        lambda c: c.get('nearest_patrol_distance_m') is not None
        and c['nearest_patrol_distance_m'] > cut)
    for field, name in (('source_sigma', 'source_sigma'),
                        ('source_strength', 'source_strength')):
        values = [c[field] for c in conditions if c.get(field) is not None]
        if not values:
            continue
        split = statistics.median(values)
        add(name, f'<= median ({split:.3f})',
            lambda c, f=field, s=split: c.get(f) is not None and c[f] <= s)
        add(name, f'> median ({split:.3f})',
            lambda c, f=field, s=split: c.get(f) is not None and c[f] > s)
    return groups, dict(nearest_patrol_distance_median=cut)


def metric_cdf_series(rows, methods, metric):
    """Empirical CDF per method; the distributions are too skewed for a mean."""
    series = {}
    for method in methods:
        values = sorted(r[metric] for r in rows if r['method'] == method
                        and r.get(metric) is not None)
        series[method] = [(v, (i + 1) / len(values))
                          for i, v in enumerate(values)] if values else []
    return series


def error_cdf_series(rows, methods):
    return metric_cdf_series(rows, methods, 'E')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).resolve().parents[1]
    root = workspace_root()
    # Paths are workspace-relative so the report survives a relocated tree.
    parser.add_argument('--runs', type=Path,
                        default=Path('outputs/analysis_20261006/runs_evaluated.csv'),
                        help='output of build_cohort.py')
    parser.add_argument('--raw-root', type=Path, default=Path('results/old'),
                        help='tree holding seed_<n>/<method>/ run folders')
    parser.add_argument('--seeds', default='all', help="'all', '50-99', or '50,60-62'")
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--reference', default=REFERENCE_METHOD)
    parser.add_argument('--merge-evidence-methods', default=','.join(MERGED_METHODS),
                        help='methods folded into the 2x2 design; kept only as evidence')
    parser.add_argument('--output', type=Path, default=Path('outputs/analysis_20261006'))
    args = parser.parse_args()
    for name in ('runs', 'raw_root', 'output'):
        value = getattr(args, name)
        setattr(args, name, value if value.is_absolute() else root/value)

    methods = tuple(m.strip() for m in args.methods.split(',') if m.strip())
    merged = tuple(m.strip() for m in args.merge_evidence_methods.split(',') if m.strip())
    rows, seeds = load_runs(args.runs, methods, args.seeds)
    if not rows:
        parser.error('no runs matched the requested seeds and methods')
    # The merged methods never enter the statistics; they only justify the merge.
    evidence_rows, _ = load_runs(args.runs, tuple(set(methods) | set(merged)), args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)

    patrol = patrol_geometry(package, args.raw_root, seeds[0])
    resolution = float(patrol.gas['gmrf_resolution'])
    stats = method_statistics(rows, methods)
    stage_rows = stage_metrics(rows, args.raw_root, patrol)
    evidence_stage_rows = stage_metrics(evidence_rows, args.raw_root, patrol)
    stages = stage_summary(stage_rows, methods)
    paired = paired_differences(rows, args.reference, methods,
                                PRIMARY_METRICS + SECONDARY_METRICS,
                                censored_metrics=CENSORED_METRICS)
    # The design's own four contrasts, Holm-corrected over just that family.
    pairs = tuple(p for p in DESIGN_PAIRS
                  if p[0] in methods and p[1] in methods)
    design_pairs = paired_comparisons(rows, pairs,
                                      PRIMARY_METRICS + SECONDARY_METRICS,
                                      censored_metrics=CENSORED_METRICS)
    scores = weighted_scores(rows, methods)
    stability = rank_stability(scores)
    completion = completion_rates(rows, methods)
    agreement = first_cell_agreement(evidence_stage_rows, resolution)
    conditions = seed_conditions(rows, patrol, args.raw_root, methods, resolution)
    bins, cuts = condition_bins(rows, conditions, methods)
    cdf = error_cdf_series(rows, methods)
    decision_cdf = metric_cdf_series(rows, methods, 'T_decision')

    write_csv(args.output/'method_stats.csv', stats)
    write_csv(args.output/'paired_vs_reference.csv', paired)
    write_csv(args.output/'paired_design_contrasts.csv', design_pairs)
    write_csv(args.output/'stage_metrics_runs.csv', stage_rows)
    write_csv(args.output/'stage_metrics.csv', stages)
    write_csv(args.output/'completion_rates.csv', completion)
    write_csv(args.output/'first_cell_agreement.csv', agreement)
    write_csv(args.output/'seed_conditions.csv', conditions)
    write_csv(args.output/'condition_bins.csv', bins)
    write_csv(args.output/'error_cdf.csv',
              [dict(method=m, error_m=v, cumulative_fraction=f)
               for m, points in cdf.items() for v, f in points])
    write_csv(args.output/'time_error_runs.csv',
              [dict(seed=r['seed'], method=r['method'],
                    T_decision=r['T_decision'], T_approach=r['T_approach'],
                    approach_succeeded=r['approach_succeeded'],
                    T=r['T'], E=r['E'], E_raw=r['E_raw'], D=r['D'])
               for r in rows])
    write_csv(args.output/'decision_time_cdf.csv',
              [dict(method=m, seconds=v, cumulative_fraction=f)
               for m, points in decision_cdf.items() for v, f in points])

    figures = dict(
        error_cdf=str(error_cdf(rows, methods, args.output/'error_cdf.png')),
        decision_time_cdf=str(decision_time_cdf(
            rows, methods, args.output/'decision_time_cdf.png')),
        time_error=str(time_error_panels(
            rows, methods, args.output/'time_error.png')),
        stage_decomposition=str(stage_decomposition(
            stages, methods, args.output/'stage_decomposition.png')))

    report = dict(
        schema_version=2, figures={k: relative(v, root) for k, v in figures.items()},
        design={m: DESIGN[m] for m in methods if m in DESIGN},
        merged_methods={m: MERGED_METHODS[m] for m in merged if m in MERGED_METHODS},
        inputs={k: relative(v, root) for k, v in vars(args).items()
                if isinstance(v, Path)},
        seeds=seeds, seed_count=len(seeds), methods=list(methods),
        reference_method=args.reference, run_count=len(rows),
        patrol_points=patrol.patrol_points,
        omitted_patrol_clusters=patrol.omitted_clusters,
        gmrf_resolution=resolution,
        condition_cuts=cuts,
        method_stats=stats, stage_summary=stages, paired_vs_reference=paired,
        design_pairs=[list(p) for p in pairs],
        paired_design_contrasts=design_pairs,
        primary_metrics=list(PRIMARY_METRICS),
        secondary_metrics=list(SECONDARY_METRICS),
        weighted_scores=scores, rank_stability=stability,
        completion_rates=completion,
        first_cell_agreement=agreement, condition_bins=bins,
        source_truth_recovery=dict(
            recovered=sum(c.get('source_truth_status') == 'recovered_from_console_log'
                          for c in conditions),
            total=len(conditions),
            coordinate_matches=sum(bool(c.get('source_log_matches_runs_csv'))
                                   for c in conditions)),
        notes=dict(
            verdict_rule=VERDICT_RULE,
            censoring=('a run whose final approach failed has no usable arrival time; '
                       'such pairs are scored as losses and left out of the magnitude '
                       'statistics, and the error of those runs is still valid'),
            design=('2x2: start position x search strategy. M3/M4 are folded into '
                    'M1/M2 on the first-cell evidence and appear only in '
                    'first_cell_agreement'),
            source_truth=('sigma and strength recovered from console.log; runs.csv '
                          'schema <= 5 omits them'),
            reporting='medians and interquartile ranges; means are reported alongside, not alone'))
    (args.output/'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(dict(seeds=len(seeds), runs=len(rows), methods=list(methods),
                          reference=args.reference,
                          navigation_failures=sum(r['navigation_failed'] for r in rows),
                          output=relative(args.output, root),
                          rank_stability=stability['identical_to_reference']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
