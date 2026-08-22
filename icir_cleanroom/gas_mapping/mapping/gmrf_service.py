"""Transactional GaBP update, retry, rollback, and CG comparison."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GmrfUpdateResult:
    success: bool
    iterations: int
    residual: float
    damping: float
    cg_max_error: float | None


class GmrfService:
    def __init__(self, config, gmrf=None):
        self.config = config
        self.gmrf = gmrf

    def attach(self, gmrf):
        self.gmrf = gmrf

    def compare_with_cg(self):
        if self.gmrf is None:
            raise RuntimeError('GMRF is not attached')
        reference, info = self.gmrf.cg_reference(
            tolerance=float(self.config.gabp_tolerance))
        if reference is None:
            return math.nan, int(info)
        maximum = float(np.max(np.abs(self.gmrf.solution - reference)))
        return maximum, 0

    def update(self, compare_with_cg=True):
        if self.gmrf is None:
            raise RuntimeError('GMRF is not attached')
        gmrf = self.gmrf
        snapshot = (
            gmrf.solution.copy(), gmrf.variance.copy(),
            gmrf.message_precision.copy(), gmrf.message_information.copy())
        damping = float(self.config.gabp_damping)
        converged, iterations, residual = gmrf.solve_gabp(
            int(self.config.gabp_max_iterations),
            float(self.config.gabp_tolerance), damping)
        if not converged:
            damping = float(self.config.gabp_retry_damping)
            gmrf.reset_gabp_messages()
            converged, iterations, residual = gmrf.solve_gabp(
                int(self.config.gabp_max_iterations),
                float(self.config.gabp_tolerance), damping)
        if not converged:
            (gmrf.solution, gmrf.variance, gmrf.message_precision,
             gmrf.message_information) = snapshot
            return GmrfUpdateResult(
                False, int(iterations), float(residual), damping, None)
        maximum = None
        if compare_with_cg:
            maximum, _ = self.compare_with_cg()
        return GmrfUpdateResult(
            True, int(iterations), float(residual), damping, maximum)


__all__ = ['GmrfService', 'GmrfUpdateResult']
