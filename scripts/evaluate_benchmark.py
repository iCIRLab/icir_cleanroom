#!/usr/bin/env python3
"""Re-evaluate a frozen cohort. Never modify input runs or launch a simulation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from icir_cleanroom.gas_mapping.application.benchmark import EvaluationDomain, cost, score
from icir_cleanroom.gas_mapping.mapping.domains import field_geometry, sampling_mask, navigation_goal_mask
from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.navigation_profile import load_navigation_settings, navigation_goal_clearance


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def restore_times(raw, lrs, events):
    """Restore ROS boundaries; legacy wall clocks have no shared monotonic origin."""
    hrs = [e for e in events if e['run_id'] == raw['run_id']]
    starts = [e for e in hrs if e['event'] == 'lrs_complete_hrs_start']
    ends = [e for e in hrs if e['event'] == 'hrs_end']
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError('missing or duplicate HRS boundary')
    start, end = float(starts[0]['ros_seconds']), float(ends[0]['ros_seconds'])
    assert math.isclose(end-start, float(raw['total_seconds']), abs_tol=1e-7)
    timestamps = [float(e['ros_seconds']) for e in hrs]
    if timestamps != sorted(timestamps) or raw['ros_clock_valid'] != 'True':
        raise ValueError('invalid HRS clock')
    laps = [r for r in lrs if r['event_id'] == raw['event_id']]
    laps.sort(key=lambda r: int(r['lrs_lap']))
    if not laps or laps[-1]['lrs_run_id'] != raw['lrs_run_id']:
        raise ValueError('missing LRS link')
    pairs = [(float(r['start_ros_seconds']), float(r['end_ros_seconds'])) for r in laps]
    if any(b<a for a,b in pairs) or any(pairs[i][0]<pairs[i-1][1] for i in range(1,len(pairs))) or pairs[-1][1]>start:
        raise ValueError('invalid LRS clock boundaries')
    return dict(
        lrs_seconds=pairs[-1][1]-pairs[-1][0],
        cumulative_lrs_seconds=sum(b-a for a,b in pairs),
        experiment_seconds=end-pairs[0][0],
        transition_seconds=start-pairs[-1][1],
        wall_lrs_seconds=float(laps[-1]['wall_total_seconds']),
        wall_cumulative_lrs_seconds=sum(float(r['wall_total_seconds']) for r in laps),
        wall_experiment_seconds=(float(raw['wall_experiment_seconds'])
                                 if raw.get('wall_experiment_seconds') else None),
        wall_experiment_status='recorded' if raw.get('wall_experiment_seconds') else 'legacy_monotonic_boundaries_missing')


def restore_domain(config, seed_folder, package):
    profile = config['profile']
    map_yaml = package/profile['environment']['map']
    meta = yaml.safe_load(map_yaml.read_text())
    if float(meta['origin'][2]) != 0:
        raise ValueError('historical map reconstruction requires zero map yaw')
    image = map_yaml.parent/meta['image']
    pixels = np.asarray(Image.open(image).convert('L'))
    probability = (pixels.astype(float) if meta['negate'] else 255.-pixels.astype(float))/255.
    occupancy = np.flipud(np.where(probability > meta['occupied_thresh'], 100,
                                  np.where(probability < meta['free_thresh'], 0, -1)))
    msg = NS(data=occupancy.ravel(), info=NS(width=occupancy.shape[1], height=occupancy.shape[0],
        resolution=float(np.float32(meta['resolution'])),
        origin=NS(position=NS(x=meta['origin'][0], y=meta['origin'][1]))))
    env = profile['gas_environment']
    nav, _ = load_navigation_settings(seed_folder/'snapshot/nav2_params.yaml', seed_folder/'snapshot/navigation_profile.yaml')
    traversal = sampling_mask(msg, env['sampling_clearance'], env['robot_start_x'], env['robot_start_y'])
    goals = navigation_goal_mask(msg, traversal, navigation_goal_clearance(nav))
    field = field_geometry(*[env[k] for k in ['map_min_x','map_max_x','map_min_y','map_max_y','gmrf_resolution']])
    domain = EvaluationDomain.snapshot(GridGeometry.from_message(msg), goals, field)
    provenance = {str(p): digest(p) for p in [map_yaml,image,seed_folder/'snapshot/nav2_params.yaml',seed_folder/'snapshot/navigation_profile.yaml']}
    return domain, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[3]
    parser.add_argument('--results', type=Path, default=workspace/'results')
    parser.add_argument('--cohort', type=Path, default=workspace/'outputs/seed_evaluation_20260921/computed_summary.json')
    parser.add_argument('--output', type=Path, default=workspace/'outputs/benchmark_20260929')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(args.cohort.read_text())
    methods = ['M'+str(i) for i in range(1,8)]
    assert len(cohort['eligible']) == 45 and len(cohort['rows']) == 315
    summary = read_csv(args.results/'summary.csv')
    lookup = {(int(r['seed']),r['method_id']):r for r in summary}
    previous_reachability_path = workspace/'outputs/source_reachability_20260921/reachability.json'
    previous_reachability = {r['seed']:r for r in json.loads(previous_reachability_path.read_text())['rows']}
    domains, all_rows, history_rows, issues = {}, [], [], []
    provenance = {str(args.cohort):digest(args.cohort), str(args.results/'summary.csv'):digest(args.results/'summary.csv'), str(previous_reachability_path):digest(previous_reachability_path)}
    conditions = {}
    for selected in cohort['rows']:
        seed, method = selected['seed'], selected['method']
        chosen = lookup[seed,method]
        path = Path(chosen['runs_csv'])
        assert path.parent.name == selected['runId']
        rows = read_csv(path)
        assert len(rows) == 1
        raw = rows[0]
        assert raw['time_basis'] == 'ros_simulation'
        assert math.isclose(float(raw['total_seconds']), selected['T'], abs_tol=1e-7)
        assert math.isclose(float(raw['source_position_error']), selected['E'], abs_tol=1e-7)
        config_path = args.results/f'seed_{seed}'/method/'run_config.yaml'
        config = yaml.safe_load(config_path.read_text())
        provenance[str(config_path)] = digest(config_path)
        parameters = json.loads(raw['parameters'])
        condition = dict(lrs=config['profile']['lrs'], gas=config['profile']['gas_environment'],
                         lrs_control={k:v for k,v in parameters.items() if k.startswith('lrs_') or k in ('load_history','save_history','hazard_threshold')})
        conditions.setdefault(seed,{})[method] = condition
        result = dict(seed=seed, method=method, run_id=raw['run_id'],
            T=float(raw['total_seconds']), E_raw=float(raw['source_position_error']),
            D=float(raw['hrs_distance_m']), wall_hrs_seconds=float(raw['wall_total_seconds']),
            source_x=float(raw['source_x']), source_y=float(raw['source_y']),
            estimated_x=float(raw['estimated_source_x']), estimated_y=float(raw['estimated_source_y']),
            classification=chosen['classification'], schema_version=raw['schema_version'])
        try:
            # Check all configuration maps, not just one method per seed.
            cache_key = (str(config['profile']['environment']['map']), json.dumps(condition['gas'],sort_keys=True),
                         digest(config_path.parent.parent/'snapshot/nav2_params.yaml'),
                         digest(config_path.parent.parent/'snapshot/navigation_profile.yaml'))
            if cache_key not in domains:
                domains[cache_key], hashes = restore_domain(config,config_path.parent.parent,Path(__file__).resolve().parents[1])
                provenance.update(hashes)
            domain = domains[cache_key]
            source = (result['source_x'],result['source_y'])
            estimate = (result['estimated_x'],result['estimated_y'])
            reference = domain.reference(source)
            old_reach = previous_reachability[seed]
            assert reference.adjusted == (not old_reach['navigation_goal_allowed'])
            assert math.isclose(reference.distance,old_reach['nearest_goal_distance'] if reference.adjusted else 0,abs_tol=1e-7)
            evaluation = reference.evaluate(estimate,source)
            result.update(evaluation)
            assert evaluation['evaluation_available']
            # Independently enumerate cell distances using NumPy rather than the reference evaluator.
            points = np.asarray(domain.candidates)
            source_distances = np.hypot(points[:,0]-source[0],points[:,1]-source[1])
            targets = points[np.abs(source_distances-source_distances.min())<=1e-9] if reference.adjusted else np.asarray([source])
            independent_error = float(np.hypot(targets[:,0]-estimate[0],targets[:,1]-estimate[1]).min())
            assert math.isclose(independent_error,evaluation['adjusted_source_position_error'],abs_tol=1e-9)
            result['E'] = independent_error
            result['cost'] = cost(result['T'], result['E'])
            history_path = path.with_name('estimate_history.csv')
            history = read_csv(history_path)
            for h in history:
                xy = ((float(h['estimated_source_x']),float(h['estimated_source_y'])) if h['estimated_source_x'] else None)
                src = ((float(h['source_x']),float(h['source_y'])) if h['source_x'] else None)
                ev = reference.evaluate(xy, src)
                history_rows.append(dict(seed=seed,method=method,**h,**ev))
            assert math.isclose(float(history_rows[-1]['adjusted_source_position_error']), result['E'],abs_tol=1e-9)
        except (OSError,ValueError,KeyError,AssertionError) as error:
            result.update(E=None,cost=None,evaluation_available=False,evaluation_reason=f'reconstruction_failed:{type(error).__name__}:{error}')
            issues.append(dict(seed=seed,method=method,reason=result['evaluation_reason']))
        try:
            result.update(restore_times(raw,read_csv(path.with_name('lrs_runs.csv')),read_csv(path.with_name('events.csv'))))
        except (OSError,ValueError,KeyError,AssertionError) as error:
            result.update(lrs_seconds=None,cumulative_lrs_seconds=None,experiment_seconds=None,wall_experiment_seconds=None,
                          wall_experiment_status='unavailable',timing_reason=str(error))
            issues.append(dict(seed=seed,method=method,reason=f'timing:{error}'))
        for p in [path,path.with_name('estimate_history.csv'),path.with_name('events.csv'),path.with_name('lrs_runs.csv')]:
            provenance[str(p)] = digest(p)
        all_rows.append(result)
    def mean(values):
        v=list(values)
        return statistics.mean(v) if all(x is not None for x in v) else None
    stats=[]
    statuses=[]
    for r in summary:
        statuses.append(dict(seed=int(r['seed']),method=r['method_id'],classification=r['classification'],runner_status=r['runner_status'],
                             failure=int(r['classification']!='normal' or r['runner_status']!='completed'),included=int(int(r['seed']) in cohort['eligible'])))
    for method in methods:
        rows=[r for r in all_rows if r['method']==method]
        costs=[r['cost'] for r in rows]
        failures=[r for r in statuses if r['method']==method]
        st=dict(method=method,n=len(rows),valid=sum(c is not None for c in costs),
            mean_cost=mean(costs),score=score(costs) if all(c is not None for c in costs) else None,
            mean_T=mean(r['T'] for r in rows),mean_E_raw=mean(r['E_raw'] for r in rows),mean_E=mean(r['E'] for r in rows),
            mean_lrs=mean(r['cumulative_lrs_seconds'] for r in rows),mean_total=mean(r['experiment_seconds'] for r in rows),
            failures=sum(r['failure'] for r in failures),all_n=len(failures))
        for weight,label in [(.5,'55'),(.4,'46')]:
            st['score_'+label]=score(cost(r['T'],r['E'],weight) for r in rows) if st['valid']==len(rows) else None
        stats.append(st)
    for key in ['score','score_55','score_46']:
        for st in stats:
            st['rank_'+key]=1+sum(x[key]>st[key]+1e-10 for x in stats if x[key] is not None) if st[key] is not None else None
    equal_conditions=all(all(v==next(iter(group.values())) for v in group.values()) for group in conditions.values())
    report=dict(schema_version=1,time_reference_seconds=160,error_reference_m=.5,time_weight=.6,error_weight=.4,
        cohort_seeds=cohort['eligible'],excluded_seeds=cohort['excluded'],rows=all_rows,stats=stats,statuses=statuses,issues=issues,
        checks=dict(independent_error_checks=sum(r['E'] is not None for r in all_rows),same_lrs_settings_within_seed=equal_conditions,
            hrs_time_median=statistics.median(r['T'] for r in all_rows),
            source_projection_count=sum(bool(r.get('evaluation_reference_adjusted')) for r in all_rows),
            historical_map_check='current map checked against prior per-seed reachability records; original run did not archive map image',
            initial_observations='identical observations not established; LRS executions are independent',
            legacy_wall_total='not reconstructed without common monotonic clock origin'))
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    write_csv(args.output/'evaluated_runs.csv',all_rows)
    write_csv(args.output/'evaluated_estimate_history.csv',history_rows)
    write_csv(args.output/'method_summary.csv',stats)
    write_csv(args.output/'selected_statuses.csv',statuses)
    print(json.dumps({'checks':report['checks'],'issues':issues,'stats':stats},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
