from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.constraint import Operator, Constraint, Constraints
import numpy as np
from scripts.drawer import draw
from scripts.utils.logger import Logger


class Simplex(BenchmarkModel):

    def _bounds(self, i, d):
        return -1, 2 * self.k + d

    def __init__(self, i, d=2.7, rows=1000):
        super().__init__(i=i, d=d, rows=rows, name='simplex')

        self.constraint_sets = [
            Constraints(constraints=[Constraint(_operator=Operator.lt, value=2 * j - 2 - self.L * (1 - bj)),
                                     Constraint(_operator=Operator.lt, value=2 * j - 2 - self.L * (1 - bj)),
                                     Constraint(_operator=Operator.gt, value=j * d + self.L * (1 - bj))]) for j, bj in
            enumerate(self.B, start=1)]

        self.bounds = [self._bounds(i=ii, d=d) for ii in self.variables]

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

    def optimal_bounding_sphere(self):
        # #FIXME: Add handling for k
        k = self.k - 1
        w = list()
        for i, j in zip(range(self.i), range(1, self.i)):
            wi = np.repeat(np.inf, self.i)
            wj = wi

            wi[i] = - Simplex.tan_pi_12(1)
            wi[j] = Simplex.cot_pi_12(1)

            wj[i] = Simplex.cot_pi_12(1)
            wj[j] = - Simplex.tan_pi_12(1)

            w.append(wi)
            w.append(wj)
        w.append(np.array(self.d))
        w = np.concatenate(w)
        w = 1 / w
        return w

    def optimal_w0(self):
        # # FIXME: Add handling for k
        # k = self.k - 1
        return np.append(np.zeros(self.optimal_n_constraints() - 1), 1)

    def optimal_n_constraints(self):
        return 2 * (self.i - 1) + 1


cube = Simplex(i=3, d=2.7)
sphere = cube.optimal_bounding_sphere()
optimal_w0 = cube.optimal_w0()
n = cube.optimal_n_constraints()
