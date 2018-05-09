from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.constraint import Operator, Constraint, Constraints
from scripts.drawer import draw
import numpy as np


class Cube(BenchmarkModel):

    def _bounds(self, i, d):
        return i - i * d * self.k, i + 2 * i * d * self.k

    def __init__(self, i, d=2.7, B=list([1, 1]), train_rows=5000, test_rows=int(1e5)):
        super().__init__(i=i, d=d, train_rows=train_rows, name='cube', test_rows=test_rows, B=B)
        self.constraint_sets = list()
        for j, bj in enumerate(self.B, start=1):
            self.constraint_sets.append([
                Constraints(constraints=[Constraint(_operator=Operator.lt, value=it * j - self.L + self.L * bj),
                                         Constraint(_operator=Operator.gt,
                                                    value=it * j + it * d + self.L - self.L * bj)])
                for it in self.variables])

        self.bounds = [self._bounds(i=ii, d=d) for ii in self.variables]

    def generate_points_from_range(self):
        df = self.generate_train_dataset()
        draw.draw2d(df=df, selected=[1, 0])
        draw.draw3d(df=df, selected=[0, 1, 2])

        self.info(df=df)

    def matches_constraints(self, row):
        validation_result = [self.match(constraints, row) for constraints in self.constraint_sets]
        return sum(validation_result) >= 1

    def match(self, constraints, row):
        for index, xi in enumerate(row):
            if not constraints[index].validate(xi):
                return False
        return True
