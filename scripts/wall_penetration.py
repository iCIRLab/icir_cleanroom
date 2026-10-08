#!/usr/bin/env python3
"""How often a hazard reading in cleanroom_amc comes through a wall.

The gas model is a static Gaussian in (x, y): it ignores geometry, so a
source can raise a patrol point above the hazard threshold with a wall in
between. This quantifies that over the random source distribution the
campaign actually draws from, and writes its own output folder.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

# The only interior partition: everything else in the layout is free-standing
# process equipment inside one open bay.
AIRLOCK = {'name': 'airlock', 'x': (-10.0, -6.5), 'y': (-7.0, -3.0)}


def rectangles(geometry):
    out = []
    for item in geometry['obstacles']:
        (cx, cy, _), (sx, sy, sz) = item['center'], item['size']
        out.append({'name': item['name'], 'height': sz,
                    'x0': cx - sx/2, 'x1': cx + sx/2,
                    'y0': cy - sy/2, 'y1': cy + sy/2})
    return out


def segment_hits_rect(p, q, rect):
    """Liang-Barsky: does segment p->q touch the axis-aligned rectangle?"""
    dx, dy = q[0] - p[0], q[1] - p[1]
    lo, hi = 0.0, 1.0
    for delta, start, low, high in (
            (-dx, -(rect['x0'] - p[0]), None, None),
            (dx, rect['x1'] - p[0], None, None),
            (-dy, -(rect['y0'] - p[1]), None, None),
            (dy, rect['y1'] - p[1], None, None)):
        if delta == 0:
            if start < 0:
                return False
            continue
        t = start/delta
        if delta < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
        if lo > hi:
            return False
    return True


def region_of(x, y):
    return (AIRLOCK['name']
            if AIRLOCK['x'][0] <= x <= AIRLOCK['x'][1]
            and AIRLOCK['y'][0] <= y <= AIRLOCK['y'][1] else 'main_bay')


def concentration(source, x, y):
    distance_sq = (x - source['x'])**2 + (y - source['y'])**2
    return source['strength']*math.exp(-distance_sq/(2.0*source['sigma']**2))


def axis(minimum, maximum, step):
    count = int(round((maximum - minimum)/step))
    return [minimum + i*step for i in range(count + 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).resolve().parents[1]
    parser.add_argument('--geometry', type=Path,
                        default=package/'config/cleanroom_amc/geometry.json')
    parser.add_argument('--patrol-points', type=Path, required=True,
                        help='JSON list of [x, y] LRS representatives')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hazard-threshold', type=float, default=0.15)
    parser.add_argument('--detection-threshold', type=float, default=0.2)
    parser.add_argument('--samples', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    geometry = json.loads(args.geometry.read_text())
    obstacles = rectangles(geometry)
    partitions = [r for r in obstacles
                  if r['name'].startswith(('wall_', 'airlock_'))]
    equipment = [r for r in obstacles if r not in partitions]
    patrol = [tuple(p) for p in json.loads(args.patrol_points.read_text())]

    rng = random.Random(args.seed)
    x_centers, y_centers = axis(-10.0, 10.0, 0.5), axis(-7.0, 7.0, 0.5)

    accepted = 0
    proposals = 0
    hazard_runs = 0
    cross_partition_runs = 0
    blocked_only_runs = 0
    first_cross_partition = 0
    first_blocked = 0
    per_point = [dict(point=p, region=region_of(*p), hazard=0,
                      cross_partition=0, blocked=0) for p in patrol]
    examples = []

    while accepted < args.samples:
        proposals += 1
        source = {'x': rng.choice(x_centers), 'y': rng.choice(y_centers),
                  'strength': rng.uniform(0.5, 1.0),
                  'sigma': rng.uniform(1.5, 3.0)}
        values = [concentration(source, *p) for p in patrol]
        if max(values) < args.detection_threshold:
            continue  # the environment node rejects and redraws
        accepted += 1
        source_region = region_of(source['x'], source['y'])
        hazards = [i for i, v in enumerate(values) if v >= args.hazard_threshold]
        if not hazards:
            continue
        hazard_runs += 1
        run_cross = run_blocked = False
        first = None
        for i in hazards:
            point = patrol[i]
            cross = region_of(*point) != source_region
            blocked = any(segment_hits_rect((source['x'], source['y']), point, r)
                          for r in obstacles)
            per_point[i]['hazard'] += 1
            per_point[i]['cross_partition'] += int(cross)
            per_point[i]['blocked'] += int(blocked)
            run_cross = run_cross or cross
            run_blocked = run_blocked or blocked
            if first is None:
                first = (cross, blocked)
            if cross and len(examples) < 20:
                examples.append({
                    'source': {k: round(v, 3) for k, v in source.items()},
                    'source_region': source_region,
                    'patrol_point': [round(c, 2) for c in point],
                    'patrol_region': region_of(*point),
                    'concentration': round(values[i], 4)})
        cross_partition_runs += int(run_cross)
        blocked_only_runs += int(run_blocked and not run_cross)
        first_cross_partition += int(bool(first and first[0]))
        first_blocked += int(bool(first and first[1]))

    def rate(count, total):
        return count/total if total else 0.0

    report = {
        'method': 'Monte Carlo over the campaign source distribution',
        'samples_accepted': accepted, 'proposals_drawn': proposals,
        'acceptance_rate': rate(accepted, proposals),
        'hazard_threshold': args.hazard_threshold,
        'detection_threshold': args.detection_threshold,
        'patrol_point_count': len(patrol),
        'partitions': {
            'interior_partition_walls': [r['name'] for r in partitions
                                         if r['name'].startswith('airlock')],
            'outer_walls': [r['name'] for r in partitions
                            if r['name'].startswith('wall_')],
            'free_standing_equipment': [r['name'] for r in equipment],
            'regions': ['airlock (x -10..-6.5, y -7..-3)', 'main_bay (rest)'],
            'patrol_points_per_region': {
                region: sum(1 for p in per_point if p['region'] == region)
                for region in ('airlock', 'main_bay')}},
        'runs_with_any_hazard': hazard_runs,
        'hazard_rate': rate(hazard_runs, accepted),
        'runs_with_cross_partition_hazard': cross_partition_runs,
        'cross_partition_rate_of_hazard_runs': rate(cross_partition_runs, hazard_runs),
        'cross_partition_rate_of_all_runs': rate(cross_partition_runs, accepted),
        'runs_with_blocked_line_of_sight_only': blocked_only_runs,
        'blocked_line_of_sight_rate_of_hazard_runs':
            rate(blocked_only_runs + cross_partition_runs, hazard_runs),
        'first_hazard_cross_partition': first_cross_partition,
        'first_hazard_cross_partition_rate': rate(first_cross_partition, hazard_runs),
        'first_hazard_line_of_sight_blocked': first_blocked,
        'first_hazard_line_of_sight_blocked_rate': rate(first_blocked, hazard_runs),
        'examples': examples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'wall_penetration.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with (args.output/'per_patrol_point.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['patrol_index', 'x', 'y', 'region', 'hazard_runs',
                         'cross_partition', 'blocked_line_of_sight',
                         'cross_partition_rate', 'blocked_rate'])
        for index, row in enumerate(per_point):
            writer.writerow([
                index, round(row['point'][0], 2), round(row['point'][1], 2),
                row['region'], row['hazard'], row['cross_partition'],
                row['blocked'], round(rate(row['cross_partition'], row['hazard']), 4),
                round(rate(row['blocked'], row['hazard']), 4)])
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
