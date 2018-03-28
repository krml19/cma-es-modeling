from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.constraint import Operator, Constraint, Constraints
from scripts.drawer import draw


class Cube(BenchmarkModel):

    def bounds(self, i, d):
        return i - i * d, i + 2 * i * d

    def __init__(self, i, d=2.7, rows=1000):
        super().__init__(i=i, d=d, rows=rows)
        self.name = 'cube'

        self.constraints = [Constraints(constraints=[Constraint(_operator=Operator.lt, value=it),
                                                     Constraint(_operator=Operator.gt, value=it + it * d)])
                            for it in self.variables]

        self.bounds = [self.bounds(i=ii, d=d) for ii in self.variables]

    def generate_points_from_range(self):
        df = self.generate_df()
        draw.draw2d(df=df, selected=[1, 0])
        draw.draw3d(df=df, selected=[0, 1, 2])

        self.info(df=df)

    def matches_constraints(self, row):
        for index, item in enumerate(row):
            if not self.constraints[index].validate(item):
                return False
        return True







