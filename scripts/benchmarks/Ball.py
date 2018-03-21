from scripts.benchmarks.BenchmarkModel import BenchmarkModel
from scripts.model.constraint import Operator, Constraint, Constraints
from scripts.drawer import draw

class Ball(BenchmarkModel):

    def bounds(self, i, d):
        return i - 2 * d, i + 2 * d

    def __init__(self, i, d=2.7):
        super().__init__(i=i, d=d)

        self.constraint = Constraints(constraints=[Constraint(_operator=Operator.gt, value=d*d)])

        self.bounds = [self.bounds(i=ii, d=d) for ii in self.variables]

    def matches_constraints(self, row):
        _sum = sum([(x-(i+1))**2 for i, x in enumerate(row)])
        return self.constraint.validate(_sum)

    def generate_points_from_range(self):
        df = self.generate_df()
        draw.draw2d(df=df, selected=[1, 0])
        draw.draw3d(df=df, selected=[0, 1, 2])
        self.info(df=df)


# ball = Ball(i=2, d=2.7)
# ball.generate_points_from_range()