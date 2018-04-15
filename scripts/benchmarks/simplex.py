from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.constraint import Operator, Constraint, Constraints
import numpy as np
from scripts.drawer import draw


class Simplex(BenchmarkModel):

    def bounds(self, i, d):
        return -1, 2 * self.k + d

    def __init__(self, i, d=2.7, rows=1000):
        super().__init__(i=i, d=d, rows=rows, name='simplex')

        self.constraint_sets = [
            Constraints(constraints=[Constraint(_operator=Operator.lt, value=2 * j - 2 - self.L * (1 - bj)),
                                     Constraint(_operator=Operator.lt, value=2 * j - 2 - self.L * (1 - bj)),
                                     Constraint(_operator=Operator.gt, value=j * d + self.L * (1 - bj))]) for j, bj in
            enumerate(self.B, start=1)]

        self.bounds = [self.bounds(i=ii, d=d) for ii in self.variables]

    def generate_points_from_range(self):
        df = self.generate_df()
        draw.draw2d(df=df, selected=[1, 0])
        draw.draw3d(df=df, selected=[0, 1, 2])
        self.info(df=df)

    def matches_constraints(self, row):
        validation_result = [self._matches_constraints_set(constraints=constraints, row=row) for constraints in
                             self.constraint_sets]
        return sum(validation_result) >= 1

    def _matches_constraints_set(self, constraints, row):
        _sum = sum(row)
        if not constraints.constraints[2].match(_sum):
            return False

        for xi, xj in zip(row.values[::1], row.values[1::1]):
            if not constraints.constraints[0].match(Simplex.constraint1(xi, xj)):
                return False
            if not constraints.constraints[1].match(Simplex.constraint2(xi, xj)):
                return False
        return True

    @staticmethod
    def tan_pi_12(x):
        return x * np.tan(np.pi / 12)

    @staticmethod
    def cot_pi_12(x):
        return x / (np.tan(np.pi / 12))

    @staticmethod
    def constraint1(xi, xj):
        return Simplex.cot_pi_12(xi) - Simplex.tan_pi_12(xj)

    @staticmethod
    def constraint2(xi, xj):
        return Simplex.cot_pi_12(xj) - Simplex.tan_pi_12(xi)
