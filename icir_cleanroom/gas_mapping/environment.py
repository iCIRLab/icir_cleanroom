"""Pure source-generation and source-state persistence helpers."""

import json
import math
import os
import tempfile

from .mapping.lattice import inclusive_axis

SOURCE_PARAMETER_NAMES = (
    'source_enabled', 'source_x', 'source_y',
    'source_strength', 'source_sigma',
)
RANDOM_SOURCE_PARAMETER_NAMES = (
    'source_mode', 'source_random_seed',
    'source_random_sigma_min', 'source_random_sigma_max',
    'source_random_min_separation', 'source_random_detection_threshold',
)
HOTSPOT_SOURCE_PARAMETER_NAMES = (
    'source_hotspot_centers', 'source_hotspot_weights',
    'source_hotspot_jitter_sigma',
)
SOURCE_CONFIGURATION_PARAMETER_NAMES = (
    RANDOM_SOURCE_PARAMETER_NAMES + HOTSPOT_SOURCE_PARAMETER_NAMES)
SOURCE_MODES = (
    'manual', 'random_after_peak', 'recurrent_hotspots_after_peak')
AUTOMATIC_SOURCE_MODES = SOURCE_MODES[1:]
RANDOM_SOURCE_MAX_ATTEMPTS = 1000


def validated_source_state(state, min_x, max_x, min_y, max_y):
    enabled = state['source_enabled']
    if not isinstance(enabled, bool):
        raise ValueError('source_enabled must be boolean')
    x, y = float(state['source_x']), float(state['source_y'])
    strength = float(state['source_strength'])
    sigma = float(state['source_sigma'])
    if not all(math.isfinite(value) for value in (x, y, strength, sigma)):
        raise ValueError('source position, strength, and sigma must be finite')
    if not (min_x <= x <= max_x and min_y <= y <= max_y):
        raise ValueError('source position must be inside the map')
    if not 0.0 <= strength <= 1.0:
        raise ValueError('source_strength must be in [0, 1]')
    if sigma <= 0.0:
        raise ValueError('source_sigma must be positive')
    return {
        'source_enabled': enabled,
        'source_x': x, 'source_y': y,
        'source_strength': strength, 'source_sigma': sigma,
    }


def validated_random_source_config(config):
    mode = str(config['source_mode'])
    seed = config['source_random_seed']
    if mode not in SOURCE_MODES:
        raise ValueError(
            f'source_mode must be one of {", ".join(SOURCE_MODES)}')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('source_random_seed must be an integer')
    sigma_min = float(config['source_random_sigma_min'])
    sigma_max = float(config['source_random_sigma_max'])
    min_separation = float(config['source_random_min_separation'])
    detection_threshold = float(
        config['source_random_detection_threshold'])
    numeric_values = (
        sigma_min, sigma_max, min_separation, detection_threshold)
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError('random source numeric parameters must be finite')
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError(
            'source random sigma range must satisfy 0 < min <= max')
    if min_separation < 0.0:
        raise ValueError('source_random_min_separation must be non-negative')
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError(
            'source_random_detection_threshold must be in [0, 1]')
    return {
        'source_mode': mode,
        'source_random_seed': seed,
        'source_random_sigma_min': sigma_min,
        'source_random_sigma_max': sigma_max,
        'source_random_min_separation': min_separation,
        'source_random_detection_threshold': detection_threshold,
    }


def validated_hotspot_source_config(
        config, min_x, max_x, min_y, max_y):
    raw_centers = config['source_hotspot_centers']
    raw_weights = config['source_hotspot_weights']
    if (isinstance(raw_centers, (str, bytes))
            or not hasattr(raw_centers, '__iter__')):
        raise ValueError(
            'source_hotspot_centers must be a flat numeric array')
    if (isinstance(raw_weights, (str, bytes))
            or not hasattr(raw_weights, '__iter__')):
        raise ValueError('source_hotspot_weights must be a numeric array')
    try:
        centers = [float(value) for value in raw_centers]
        weights = [float(value) for value in raw_weights]
        jitter_sigma = float(config['source_hotspot_jitter_sigma'])
    except (TypeError, ValueError) as error:
        raise ValueError(
            'hotspot centers, weights, and jitter sigma must be numeric') \
            from error
    if len(centers) < 2 or len(centers) % 2 != 0:
        raise ValueError(
            'source_hotspot_centers must contain one or more x,y pairs')
    hotspot_count = len(centers) // 2
    if len(weights) != hotspot_count:
        raise ValueError(
            'source_hotspot_weights must have one value per hotspot center')
    if not all(math.isfinite(value) for value in (
            centers + weights + [jitter_sigma])):
        raise ValueError('hotspot source numeric parameters must be finite')
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError(
            'source_hotspot_weights must be non-negative with a positive sum')
    if jitter_sigma < 0.0:
        raise ValueError('source_hotspot_jitter_sigma must be non-negative')
    for index in range(hotspot_count):
        x = centers[2 * index]
        y = centers[2 * index + 1]
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            raise ValueError(
                f'hotspot center {index} must be inside the map')
    return {
        'source_hotspot_centers': centers,
        'source_hotspot_weights': weights,
        'source_hotspot_jitter_sigma': jitter_sigma,
    }


def random_source_is_lrs_detectable(
        state, min_x, max_x, min_y, max_y, lrs_spacing,
        detection_threshold, sampling_points=None):
    points = sampling_points
    if points is None:
        points = [
            (x, y)
            for y in inclusive_axis(min_y, max_y, lrs_spacing)
            for x in inclusive_axis(min_x, max_x, lrs_spacing)]
    for x, y in points:
        distance_sq = (
            (x - state['source_x']) ** 2
            + (y - state['source_y']) ** 2)
        concentration = math.exp(
            -distance_sq / (2.0 * state['source_sigma'] ** 2))
        if concentration >= detection_threshold:
            return True
    return False


def generate_random_source_state(
        rng, min_x, max_x, min_y, max_y, gmrf_resolution,
        sigma_min, sigma_max, min_separation, lrs_spacing,
        detection_threshold, previous_position=None,
        max_attempts=RANDOM_SOURCE_MAX_ATTEMPTS, sampling_points=None):
    x_centers = inclusive_axis(min_x, max_x, gmrf_resolution)
    y_centers = inclusive_axis(min_y, max_y, gmrf_resolution)
    min_separation_sq = min_separation ** 2
    for _ in range(max_attempts):
        proposed = {
            'source_enabled': True,
            'source_x': float(rng.choice(x_centers)),
            'source_y': float(rng.choice(y_centers)),
            'source_strength': 1.0,
            'source_sigma': float(rng.uniform(sigma_min, sigma_max)),
        }
        if previous_position is not None:
            dx = proposed['source_x'] - previous_position[0]
            dy = proposed['source_y'] - previous_position[1]
            if dx * dx + dy * dy < min_separation_sq:
                continue
        if random_source_is_lrs_detectable(
                proposed, min_x, max_x, min_y, max_y, lrs_spacing,
                detection_threshold, sampling_points):
            return proposed
    raise ValueError(
        f'could not generate a detectable random gas source in '
        f'{max_attempts} attempts')


def snap_to_grid(value, minimum, resolution):
    if not math.isfinite(float(value)):
        raise ValueError('grid coordinate must be finite')
    if not math.isfinite(float(resolution)) or resolution <= 0.0:
        raise ValueError('grid resolution must be positive and finite')
    grid_index = int(math.floor(
        (float(value) - minimum) / resolution + 0.5))
    return float(minimum + grid_index * resolution)


def generate_recurrent_hotspot_source_state(
        rng, min_x, max_x, min_y, max_y, gmrf_resolution,
        sigma_min, sigma_max, lrs_spacing, detection_threshold,
        hotspot_centers, hotspot_weights, hotspot_jitter_sigma,
        max_attempts=RANDOM_SOURCE_MAX_ATTEMPTS, sampling_points=None):
    hotspots = [
        (hotspot_centers[index], hotspot_centers[index + 1])
        for index in range(0, len(hotspot_centers), 2)]
    for _ in range(max_attempts):
        hotspot_index = rng.choices(
            range(len(hotspots)), weights=hotspot_weights, k=1)[0]
        center_x, center_y = hotspots[hotspot_index]
        proposed = {
            'source_enabled': True,
            'source_x': snap_to_grid(
                rng.gauss(center_x, hotspot_jitter_sigma),
                min_x, gmrf_resolution),
            'source_y': snap_to_grid(
                rng.gauss(center_y, hotspot_jitter_sigma),
                min_y, gmrf_resolution),
            'source_strength': 1.0,
            'source_sigma': float(rng.uniform(sigma_min, sigma_max)),
        }
        if not (
                min_x <= proposed['source_x'] <= max_x
                and min_y <= proposed['source_y'] <= max_y):
            continue
        if random_source_is_lrs_detectable(
                proposed, min_x, max_x, min_y, max_y, lrs_spacing,
                detection_threshold, sampling_points):
            return proposed
    raise ValueError(
        f'could not generate a detectable recurrent-hotspot gas source in '
        f'{max_attempts} attempts')


def load_source_state(path):
    expanded = os.path.expanduser(str(path))
    if not os.path.exists(expanded):
        return None
    with open(expanded, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if int(payload.get('version', -1)) != 1:
        raise ValueError('unsupported gas source state version')
    source = payload.get('source')
    if not isinstance(source, dict) or any(
            name not in source for name in SOURCE_PARAMETER_NAMES):
        raise ValueError('gas source state is incomplete')
    return {name: source[name] for name in SOURCE_PARAMETER_NAMES}


def save_source_state(path, state):
    expanded = os.path.expanduser(str(path))
    directory = os.path.dirname(expanded) or '.'
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', dir=directory, prefix='.gas_source_state_',
                suffix='.json', delete=False, encoding='utf-8') as stream:
            json.dump({'version': 1, 'source': state}, stream, indent=2)
            temporary_path = stream.name
        os.replace(temporary_path, expanded)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
