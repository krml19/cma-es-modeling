from scripts.benchmarks.BenchmarkModel import BenchmarkModel
from scripts.model.constraint import Operator, Constraint, Constraints


class Cube(BenchmarkModel):

    def bounds(self, i, d):
        return i - i * d, i + 2 * i * d

    def __init__(self, i, d=2.7):
        super().__init__(i=i, d=d)

        self.constraints = [Constraints(constraints=[Constraint(_operator=Operator.lt, value=it),
                                                     Constraint(_operator=Operator.gt, value=it + it * d)])
                            for it in self.variables]

        self.bounds = [self.bounds(i=ii, d=d) for ii in self.variables]

    def generate_points_from_range(self):
        df = self.generate_df()
        self.draw2d(df=df, selected=[1, 0])
        self.info(df=df)

    def matches_constraints(self, row):
        for index, item in enumerate(row):
            if not self.constraints[index].validate(item):
                return False
        return True


cube = Cube(i=2, d=2.7)
cube.generate_points_from_range()







