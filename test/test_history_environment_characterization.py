"""Characterize persistence and source-environment calculations."""

import json
import random

import numpy as np
import pytest

from icir_cleanroom.gas_mapping.environment import (
    generate_random_source_state, load_source_state, save_source_state,
    validated_source_state)
from icir_cleanroom.gas_mapping.history import GasHistoryStore


def test_history_v3_round_trip_and_duplicate_event(map_message_factory, tmp_path):
    path = tmp_path / 'history.json'
    history = GasHistoryStore(path, map_message_factory(), 0.3, 0.5)
    history.update(1, 1, 0.8, 'LRS', 10.0)
    history.accumulate_distribution(
        np.linspace(0.0, 0.8, 9),
        np.asarray([(r, c) for r in range(3) for c in range(3)]))
    assert history.record_confirmed_peak('event-1', 1, 1, 0.8, 11.0)
    assert not history.record_confirmed_peak('event-1', 0, 0, 0.1, 12.0)
    history.save()

    loaded = GasHistoryStore(path, map_message_factory(), 0.3, 0.5)
    assert loaded.load()
    assert loaded.records == history.records
    np.testing.assert_array_equal(loaded.history_mean, history.history_mean)
    np.testing.assert_array_equal(loaded.history_recent, history.history_recent)
    assert loaded.confirmed_peak_events == history.confirmed_peak_events


@pytest.mark.parametrize('version', [1, 2])
def test_history_loads_legacy_versions(
        map_message_factory, tmp_path, version):
    path = tmp_path / f'history-v{version}.json'
    payload = {
        'version': version,
        'map': {'width': 3, 'height': 3, 'resolution': 1.0,
                'origin_x': 0.0, 'origin_y': 0.0},
        'records': [],
    }
    if version == 2:
        payload.update(history_count=1, history_mean=[0.2] * 9)
    path.write_text(json.dumps(payload), encoding='utf-8')

    history = GasHistoryStore(path, map_message_factory(), 0.3)
    assert history.load()
    assert history.history_count == (1 if version == 2 else 0)
    if version == 2:
        np.testing.assert_array_equal(history.history_recent, 0.2)


def test_invalid_history_does_not_partially_replace_state(
        map_message_factory, tmp_path):
    path = tmp_path / 'invalid.json'
    history = GasHistoryStore(path, map_message_factory(), 0.3)
    history.update(1, 1, 0.8, 'LRS', 10.0)
    before = dict(history.records)
    path.write_text(json.dumps({
        'version': 3,
        'map': {'width': 3, 'height': 3, 'resolution': 1.0,
                'origin_x': 0.0, 'origin_y': 0.0},
        'records': [], 'history_count': 1, 'history_mean': [0.0],
        'history_recent': [0.0] * 9,
    }), encoding='utf-8')

    with pytest.raises(ValueError):
        history.load()
    assert history.records == before


def test_source_state_round_trip_and_seeded_generation(tmp_path):
    state = validated_source_state({
        'source_enabled': True, 'source_x': 1, 'source_y': 2,
        'source_strength': 0.8, 'source_sigma': 3,
    }, -10, 10, -10, 10)
    path = tmp_path / 'source.json'
    save_source_state(path, state)
    assert load_source_state(path) == state

    first = generate_random_source_state(
        random.Random(7), -5, 5, -5, 5, 1.0,
        1.0, 2.0, 0.0, 2.0, 0.1)
    second = generate_random_source_state(
        random.Random(7), -5, 5, -5, 5, 1.0,
        1.0, 2.0, 0.0, 2.0, 0.1)
    assert first == second
