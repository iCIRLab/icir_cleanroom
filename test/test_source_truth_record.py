"""Source sigma, strength, and discarded-proposal counts reach runs.csv."""
import csv
import json
import random
from types import SimpleNamespace as NS

from icir_cleanroom.gas_mapping.application.benchmark import SOURCE_TRUTH_FIELDS
from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog
from icir_cleanroom.gas_mapping.environment import (
    generate_random_source_state, new_generation_stats)
from icir_cleanroom.gas_mapping.ros.hrs_run_logging import source_truth

GRID_POINTS = [(float(x), float(y)) for y in range(-5, 6) for x in range(-5, 6)]


def finished_row(tmp_path, truth):
    log = HrsRunLog(tmp_path, 'offline_test_clock')
    log.start(0., 0., event_id='event-1', lrs_lap=1, source=(1., 2.),
              parameters={'method': 'M3'})
    log.measurement(1., 1., xy=(1., 2.), value=.5, sample_count=4)
    return log, log.finish(2., 2., reason='done', robot_xy=(1., 2.),
                           source_end=(1., 2.), source_truth=truth)


def test_discarded_proposals_are_counted_without_changing_the_accepted_source():
    stats = new_generation_stats()
    # Only the far corner clears the threshold, so most proposals are discarded.
    accepted = generate_random_source_state(
        random.Random(3), -5., 5., -5., 5., 1.0,
        1.0, 2.0, 1.0, 1.0, 0.0, 0.1, [(5., 5.)], stats=stats)
    unlogged = generate_random_source_state(
        random.Random(3), -5., 5., -5., 5., 1.0,
        1.0, 2.0, 1.0, 1.0, 0.0, 0.1, [(5., 5.)])
    assert accepted == unlogged  # Counting must not consume extra randomness.
    assert stats['source_rejected_detection'] > 0
    assert stats['source_generation_attempts'] == stats['source_rejected_detection'] + 1
    assert stats['source_rejected_separation'] == 0
    assert stats['source_generation_exhausted'] is False


def test_separation_rejections_are_counted_separately():
    stats = new_generation_stats()
    generate_random_source_state(
        random.Random(11), -5., 5., -5., 5., 1.0,
        1.0, 2.0, 1.0, 1.0, 4.0, 0.1, GRID_POINTS,
        previous_position=(0., 0.), stats=stats)
    assert stats['source_rejected_separation'] > 0
    assert stats['source_rejected_detection'] == 0


def test_exhausted_generation_reports_counts_before_raising():
    stats = new_generation_stats()
    try:
        generate_random_source_state(
            random.Random(5), -5., 5., -5., 5., 1.0,
            1.0, 2.0, 1.0, 1.0, 0.0, 0.9, [(500., 500.)],
            max_attempts=7, stats=stats)
    except ValueError:
        pass
    else:
        raise AssertionError('undetectable source should not be accepted')
    assert stats['source_generation_attempts'] == 7
    assert stats['source_rejected_detection'] == 7
    assert stats['source_generation_exhausted'] is True


def test_runs_csv_carries_sigma_strength_and_rejection_counts(tmp_path):
    truth = dict(source_sigma=2.5, source_strength=0.75, source_mode='random_after_peak',
                 source_detection_threshold=0.2, source_detection_point_count=10,
                 source_generation_attempts=4, source_rejected_detection=3,
                 source_rejected_separation=0, source_rejected_out_of_bounds=0,
                 source_generation_exhausted=False)
    log, row = finished_row(tmp_path, truth)
    assert row['source_sigma'] == 2.5 and row['source_strength'] == 0.75
    assert row['source_rejected_detection'] == 3
    with (log.directory / 'runs.csv').open() as stream:
        saved = next(csv.DictReader(stream))
    assert saved['source_sigma'] == '2.5' and saved['source_rejected_detection'] == '3'
    assert all(field in saved for field in SOURCE_TRUTH_FIELDS)


def test_missing_source_truth_leaves_the_fields_blank(tmp_path):
    log, row = finished_row(tmp_path, None)
    assert all(row[field] is None for field in SOURCE_TRUTH_FIELDS)
    with (log.directory / 'runs.csv').open() as stream:
        saved = next(csv.DictReader(stream))
    assert all(saved[field] == '' for field in SOURCE_TRUTH_FIELDS)


def test_malformed_truth_payloads_are_discarded_rather_than_recorded():
    controller = NS(hrs_log_source_truth={'source_sigma': 1.0})
    for payload in ('not json', '[1, 2]', 'null', '"text"'):
        source_truth(controller, NS(data=payload))
        assert controller.hrs_log_source_truth is None
    source_truth(controller, NS(data=json.dumps({'source_sigma': 3.0})))
    assert controller.hrs_log_source_truth == {'source_sigma': 3.0}
