"""Characterize environment profiles and vendored world dependencies."""

from pathlib import Path
import re

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_profile(name):
    path = PACKAGE_ROOT / 'config' / 'environments' / f'{name}.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def test_environment_profiles_resolve_assets_and_aligned_field_grid():
    for name in ('empty_50m', 'aws_small_warehouse'):
        profile = load_profile(name)
        environment = profile['environment']
        assert (PACKAGE_ROOT / environment['world']).is_file()
        assert (PACKAGE_ROOT / environment['navigation_profile']).is_file()
        map_yaml = PACKAGE_ROOT / environment['map']
        assert map_yaml.is_file()
        map_config = yaml.safe_load(map_yaml.read_text(encoding='utf-8'))
        assert (map_yaml.parent / map_config['image']).is_file()

        gas = profile['gas_environment']
        assert profile['lrs']['lrs_stride_cells'] == 10
        assert 'lrs_spacing' not in profile['lrs']
        assert 'source_random_lrs_spacing' not in gas
        for resolution_name in (
                'ground_truth_resolution', 'gmrf_resolution'):
            resolution = gas[resolution_name]
            width = (gas['map_max_x'] - gas['map_min_x']) / resolution
            height = (gas['map_max_y'] - gas['map_min_y']) / resolution
            assert abs(width - round(width)) < 1.0e-9
            assert abs(height - round(height)) < 1.0e-9


def test_environment_profiles_select_the_expected_navigation_profile():
    assert load_profile('empty_50m')['environment'][
        'navigation_profile'] == 'config/navigation/empty_fast.yaml'
    assert load_profile('aws_small_warehouse')['environment'][
        'navigation_profile'] == (
            'config/navigation/aws_warehouse_safe.yaml')


def test_aws_world_model_dependencies_and_provenance_are_complete():
    world_path = (
        PACKAGE_ROOT / 'worlds' / 'aws_small_warehouse' /
        'small_warehouse.world')
    world = world_path.read_text(encoding='utf-8')
    model_names = set(re.findall(r'model://([^<]+)', world))
    assert model_names
    model_root = PACKAGE_ROOT / 'models' / 'aws_small_warehouse'
    for model_name in model_names:
        model_dir = model_root / model_name
        assert (model_dir / 'model.config').is_file()
        model_sdf = (model_dir / 'model.sdf').read_text(encoding='utf-8')
        assert 'file://models/' not in model_sdf
        for mesh in re.findall(r'<uri>model://[^/]+/([^<]+)</uri>', model_sdf):
            assert (model_dir / mesh).is_file()
