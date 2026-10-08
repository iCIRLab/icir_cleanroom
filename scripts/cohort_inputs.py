#!/usr/bin/env python3
"""Read-only loaders for a finished seed campaign. Never writes into results/."""
import csv
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import yaml
from PIL import Image

from icir_cleanroom.gas_mapping.mapping.domains import (
    field_geometry, mask_contains_world, navigation_goal_mask, sampling_mask)
from icir_cleanroom.gas_mapping.mapping.grid_geometry import GridGeometry
from icir_cleanroom.gas_mapping.mapping.lrs_representatives import (
    build_lrs_representatives)
from icir_cleanroom.gas_mapping.navigation_profile import (
    load_navigation_settings, navigation_goal_clearance)

SOURCE_LOG = re.compile(
    r'random gas source generated: position=\(([-\d.]+),([-\d.]+)\), '
    r'strength=([\d.]+), sigma=([\d.]+)')


def workspace_root():
    """The colcon workspace that holds results/ and outputs/."""
    return Path(__file__).resolve().parents[3]


def relative(path, root=None):
    """Store paths relative to the workspace so reports survive relocation."""
    root = Path(root or workspace_root())
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_seeds(text, available):
    """Accept '50-99', '50,51,60-62', or 'all'."""
    if text in (None, '', 'all'):
        return sorted(available)
    wanted = set()
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part.lstrip('-'):
            low, high = part.split('-', 1)
            wanted.update(range(int(low), int(high) + 1))
        else:
            wanted.add(int(part))
    return sorted(wanted & set(available))


def run_folder(raw_root, seed, method):
    """Locate a method's HRS output folder under a relocated results tree."""
    base = Path(raw_root) / f'seed_{seed}' / method / 'hrs'
    if not base.is_dir():
        return None
    folders = sorted(p.parent for p in base.glob('*/runs.csv'))
    return folders[-1] if folders else None


def recover_source_truth(raw_root, seed, methods):
    """Recover sigma and strength from console logs; runs.csv v<=5 omits them."""
    values = set()
    for method in methods:
        path = Path(raw_root) / f'seed_{seed}' / method / 'console.log'
        if not path.exists():
            continue
        for match in SOURCE_LOG.findall(path.read_text(errors='ignore')):
            values.add(tuple(float(v) for v in match))
    if len(values) != 1:
        return dict(source_sigma=None, source_strength=None,
                    source_truth_status='missing' if not values else 'conflicting',
                    source_truth_variants=len(values))
    x, y, strength, sigma = next(iter(values))
    return dict(source_sigma=sigma, source_strength=strength,
                source_log_x=x, source_log_y=y,
                source_truth_status='recovered_from_console_log',
                source_truth_variants=1)


def occupancy_message(map_yaml):
    meta = yaml.safe_load(Path(map_yaml).read_text())
    if float(meta['origin'][2]) != 0:
        raise ValueError('map reconstruction requires zero map yaw')
    image = Path(map_yaml).parent / meta['image']
    pixels = np.asarray(Image.open(image).convert('L')).astype(float)
    probability = (pixels if meta['negate'] else 255. - pixels) / 255.
    grid = np.flipud(np.where(probability > meta['occupied_thresh'], 100,
                              np.where(probability < meta['free_thresh'], 0, -1)))
    return NS(data=grid.ravel(), info=NS(
        width=grid.shape[1], height=grid.shape[0],
        resolution=float(np.float32(meta['resolution'])),
        origin=NS(position=NS(x=meta['origin'][0], y=meta['origin'][1])))), meta


def patrol_geometry(package, raw_root, seed, method='M1'):
    """Rebuild patrol representatives and reachable goal cells from the snapshot."""
    seed_folder = Path(raw_root) / f'seed_{seed}'
    config = yaml.safe_load((seed_folder / method / 'run_config.yaml').read_text())
    profile = config['profile']
    message, _ = occupancy_message(Path(package) / profile['environment']['map'])
    env, lrs = profile['gas_environment'], profile['lrs']
    nav, _ = load_navigation_settings(seed_folder / 'snapshot/nav2_params.yaml',
                                      seed_folder / 'snapshot/navigation_profile.yaml')
    traversal = sampling_mask(message, env['sampling_clearance'],
                              env['robot_start_x'], env['robot_start_y'])
    goals = navigation_goal_mask(message, traversal, navigation_goal_clearance(nav))
    field = field_geometry(*[env[k] for k in (
        'map_min_x', 'map_max_x', 'map_min_y', 'map_max_y', 'gmrf_resolution')])
    geometry = GridGeometry.from_message(message)
    selection = build_lrs_representatives(
        field, geometry, traversal, goals,
        lrs['lrs_cluster_count'], lrs['lrs_cluster_random_seed'])
    reachable = [geometry.cell_center(row, column)
                 for row, column in zip(*np.nonzero(goals))]
    return NS(patrol_points=[(p.x, p.y) for p in selection.representatives],
              reachable_cells=np.asarray(reachable, dtype=float),
              goal_mask=goals, goal_geometry=geometry,
              field_geometry=field,
              omitted_clusters=list(selection.omitted_cluster_ids),
              gas=env, lrs=lrs)


def source_is_reachable(patrol, source):
    """Mask lookup, not a distance test: cell centres never coincide with the source."""
    return mask_contains_world(patrol.goal_mask, patrol.goal_geometry, *source)


def nearest_patrol_distance(patrol_points, source):
    return min(math.dist(point, source) for point in patrol_points)


def patrol_peak_concentration(patrol_points, source, sigma, strength):
    """Highest concentration the patrol representatives could have seen."""
    if sigma in (None, 0) or strength is None:
        return None
    return max(strength * math.exp(-math.dist(point, source) ** 2 / (2. * sigma ** 2))
               for point in patrol_points)


def first_target_cell(events):
    for row in events:
        if row['event'] != 'target_selected':
            continue
        try:
            details = json.loads(row['details'])
            return (float(details['x']), float(details['y'])), float(row['ros_seconds'])
        except (KeyError, TypeError, ValueError):
            return None, None
    return None, None


def boundary_times(events):
    """The boundaries the time metrics are cut on, in ROS seconds.

    `final_estimate` closes the search: everything after it is the final
    approach, which is reported on its own rather than inside search time.
    Older logs have no final_approach_end event, so the approach outcome
    falls back to the termination reason at the call site.
    """
    def first(event):
        for row in events:
            if row['event'] == event:
                return float(row['ros_seconds'])
        return None

    def details(event):
        for row in events:
            if row['event'] == event:
                try:
                    return json.loads(row.get('details') or '{}')
                except ValueError:
                    return {}
        return {}
    approach = details('final_approach_end')
    return dict(hrs_start=first('lrs_complete_hrs_start'),
                first_measurement=first('measurement_complete'),
                final_estimate=first('final_estimate_selected'),
                final_approach_start=first('final_approach_start'),
                final_approach_end=first('final_approach_end'),
                final_approach_succeeded=approach.get('succeeded'),
                final_approach_reason=approach.get('reason'),
                hrs_end=first('hrs_end'))


def best_measurement_milestone(history, tolerance=1e-9):
    """ROS time when the run first measured its eventual best concentration."""
    values = [(float(r['ros_seconds']), float(r['best_measured_concentration']))
              for r in history
              if r.get('best_measured_concentration') not in (None, '')]
    if not values:
        return None, None
    best = max(value for _, value in values)
    for ros, value in values:
        if value >= best - tolerance:
            return ros, best
    return None, best
