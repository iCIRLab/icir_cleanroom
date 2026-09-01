"""Lock down the current GMRF and HRS policy behavior."""

import numpy as np

from icir_cleanroom.gas_mapping.mapping.gmrf import GmrfGrid
from icir_cleanroom.gas_mapping.planning.hrs_policy import (
    distance_discounted_ucb, normalized_ucb)


def test_observation_replaces_nearest_cell_and_reset_restores_prior(
        map_message_factory):
    gmrf = GmrfGrid(map_message_factory(), background_mean=0.1)

    assert gmrf.set_observation(1.5, 1.5, 0.8) == (1, 1)
    assert gmrf.obs_weight.tolist() == [0.0] * 4 + [125.0] + [0.0] * 4
    assert gmrf.obs_rhs.tolist() == [0.0] * 4 + [100.0] + [0.0] * 4

    gmrf.set_observation(1.5, 1.5, 0.4)
    assert gmrf.obs_weight[4] == 250.0
    assert gmrf.obs_rhs[4] == 100.0

    gmrf.reset_observations()
    np.testing.assert_array_equal(gmrf.obs_weight, 0.0)
    np.testing.assert_array_equal(gmrf.obs_rhs, 0.0)
    np.testing.assert_array_equal(gmrf.solution, 0.1)
    np.testing.assert_array_equal(gmrf.message_precision, 0.0)
    np.testing.assert_array_equal(gmrf.message_information, 0.0)


def test_gabp_matches_locked_cg_solution(map_message_factory):
    gmrf = GmrfGrid(map_message_factory())
    gmrf.set_observation(1.5, 1.5, 0.8)

    converged, iterations, residual = gmrf.solve_gabp(
        max_iterations=500, tolerance=1.0e-9, damping=0.5)
    reference, info = gmrf.cg_reference(
        max_iterations=2000, tolerance=1.0e-9)

    assert converged
    assert iterations == 51
    assert residual <= 1.0e-9
    assert info == 0
    np.testing.assert_allclose(gmrf.solution, reference, atol=1.0e-9)
    np.testing.assert_allclose(gmrf.solution, [
        0.225257955687, 0.283389041356, 0.225257955687,
        0.283389041356, 0.777503269798, 0.283389041356,
        0.225257955687, 0.283389041356, 0.225257955687,
    ], atol=1.0e-9)


def test_normalized_ucb_is_preserved():
    np.testing.assert_allclose(
        normalized_ucb([0.1, 0.8], [0.04, 0.01], 1.5),
        [0.4, 0.95])


def test_distance_discounted_ucb_matches_hrs_candidate_score():
    np.testing.assert_allclose(
        distance_discounted_ucb(
            [0.5, 0.6], [0.0, 0.0], [0.0, 2.0],
            coefficient=0.0, distance_weight=0.5),
        [0.5, 0.3])


def test_gmrf_variable_and_directed_message_order(map_message_factory):
    gmrf = GmrfGrid(map_message_factory())

    assert gmrf.var_cells.tolist() == [
        [0, 0], [0, 1], [0, 2],
        [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2],
    ]
    assert len(gmrf.message_precision) == 40
    assert list(zip(gmrf.edge_source[:6], gmrf.edge_target[:6])) == [
        (0, 1), (1, 0), (0, 3), (3, 0), (0, 4), (4, 0)]
