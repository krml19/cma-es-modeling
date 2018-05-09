from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.constraint import Operator, Constraint, Constraints
from scripts.drawer import draw
import numpy as np


class Ball(BenchmarkModel):

    def _bounds(self, i, d):
        return i - 2 * d, i + 2 * d + (2 * np.sqrt(6) * (self.k - 1) * d) / np.pi

    def __init__(self, i, d=2.7, B=list([1, 1])):
        super().__init__(i=i, d=d, name='ball', B=B)

        self.constraint_sets = [
            Constraints(constraints=[Constraint(_operator=Operator.gt, value=d * d + self.L * (1 - bj))]) for bj in
            self.B]

        self.bounds = [self._bounds(i=ii, d=d) for ii in self.variables]

    def matches_constraints(self, row):
        validation_result = [self._match_constraint_set(constraint=constraints, j=j, row=row) for j, constraints in
                             enumerate(self.constraint_sets)]
        return sum(validation_result) >= 1

    def _match_constraint_set(self, constraint, j, row):
        validation_result = [(x - i - (2 * np.sqrt(6) * j * self.d[i-1]) / (i * np.pi)) ** 2 for i, x in enumerate(row, start=1)]
        _sum = sum(validation_result)
        return constraint.validate(_sum)
