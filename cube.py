from benchmark_model import BenchmarkModel
from constraint import Operator, Constraint, Constraints


class Cube(BenchmarkModel):

    def _bounds(self, i, d):
        return i - i * d * self.k, i + 2 * i * d * self.k

    def __init__(self, i, d=2.7, B=list([1, 1])):
        super().__init__(i=i, d=d, name='cube', B=B)
        self.constraint_sets = list()
        for j, bj in enumerate(self.B, start=1):
            self.constraint_sets.append([
                Constraints(constraints=[Constraint(_operator=Operator.lt, value=it * j - self.L + self.L * bj),
                                         Constraint(_operator=Operator.gt,
                                                    value=it * j + it * d + self.L - self.L * bj)])
                for it in self.variables])

        self.bounds = [self._bounds(i=ii, d=d) for ii in self.variables]

    def matches_constraints(self, row):
        validation_result = [self.match(constraints, row) for constraints in self.constraint_sets]
        return sum(validation_result) >= 1

    def match(self, constraints, row):
        for index, xi in enumerate(row):
            if not constraints[index].validate(xi):
                return False
        return True
