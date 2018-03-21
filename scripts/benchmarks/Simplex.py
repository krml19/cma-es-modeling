from scripts.benchmarks.BenchmarkModel import BenchmarkModel
from scripts.model.constraint import Operator, Constraint, Constraints
import numpy as np


class Simplex(BenchmarkModel):

    def bounds(self, i, d):
        return -1, 2 + d

    def __init__(self, i, d=2.7):
        super().__init__(i=i, d=d)

        self.constraints = Constraints(constraints=[Constraint(_operator=Operator.lt, value=0),
                                                    Constraint(_operator=Operator.lt, value=0),
                                                    Constraint(_operator=Operator.gt, value=d)])

        self.bounds = [self.bounds(i=ii, d=d) for ii in self.variables]

    def generate_points_from_range(self):
        df = self.generate_df()
        self.draw2d(df=df, selected=[1, 0])
        self.info(df=df)

    def matches_constraints(self, row):
        _sum = sum(row)
        if not self.constraints.constraints[2].match(_sum):
            return False

        for xi, xj in zip(row.values[::1], row.values[1::1]):
            if not self.constraints.constraints[0].match(Simplex.constraint1(xi, xj)):
                return False
            if not self.constraints.constraints[1].match(Simplex.constraint2(xi, xj)):
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


model = Simplex(i=2, d=2.7)
model.generate_points_from_range()
