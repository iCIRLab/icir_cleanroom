"""Peak-confirmation state owner."""

from ..models import PeakSearchState
from ..planning.hrs import HrsCell
from ..planning.hrs_policy import higher_measured_neighbor, normalized_ucb


class PeakConfirmationManager:
    def __init__(self, state=None):
        self.state = state or PeakSearchState()

    def reset(self):
        self.state.reset()

    def begin(self, anchor_variable, trigger_reason):
        self.state.anchor_variable = int(anchor_variable)
        self.state.move_count = 0
        self.state.trigger_reason = str(trigger_reason)
        self.state.returning = False

    def move_anchor(self, variable):
        self.state.anchor_variable = int(variable)
        self.state.move_count += 1

    @staticmethod
    def build_cells(gmrf, variables, ucb_coefficient):
        potentials = normalized_ucb(
            gmrf.solution, gmrf.variance, float(ucb_coefficient))
        cells = []
        for variable in variables:
            row, col = gmrf.var_cells[variable]
            x, y = gmrf.cell_center(variable)
            cells.append(HrsCell(
                variable=int(variable), row=int(row), col=int(col),
                x=x, y=y, reward=float(potentials[variable]),
                mean=float(gmrf.solution[variable]),
                variance=float(gmrf.variance[variable])))
        cells.sort(key=lambda cell: (cell.row, cell.col))
        return cells

    @staticmethod
    def higher_neighbor(
            anchor, neighbors, measured_values, variable_cells, epsilon):
        return higher_measured_neighbor(
            anchor, neighbors, measured_values, variable_cells, epsilon)


__all__ = ['PeakConfirmationManager']
