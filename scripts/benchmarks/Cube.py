import numpy as np
from scripts.benchmarks.BenchmarkModel import BenchmarkModel
from scripts.model.constraint import Operator, Constraint
from scripts import sampler
import pandas as pd
import matplotlib.pyplot as plt


class Cube(BenchmarkModel):
    rows = 1000

    def bounds(self, i, d):
        return i - i * d, i + 2 * i * d

    def __init__(self, i, d=2.7):
        super().__init__(i=i, d=d)
        self.constraints = [[Constraint(_operator=Operator.gt, value=it),
                   Constraint(_operator=Operator.lt, value=it + it * d)] for it in self.variables]
        self.bounds = [self.bounds(i=ii, d=d) for ii in self.variables]

    def generate_points_from_range(self):
        cols = self.variable_names(self.variables)
        samples = sampler.samples(self.bounds, rows=Cube.rows, cols=len(self.variables))
        df = pd.DataFrame(samples.T, columns=cols)
        df.plot(kind='scatter', x=cols[0], y=cols[1])
        print(df.describe())
        plt.show()


cube = Cube(i=2, d=2.7)
cube.generate_points_from_range()







