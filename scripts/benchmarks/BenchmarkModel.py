import numpy as np
import matplotlib.pyplot as plt
from scripts import sampler
import pandas as pd


class BenchmarkModel:
    rows = 10000

    def __init__(self, i, d=2.7):
        self.variables = np.arange(1, i + 1)
        self.d = d * np.ones(i)

    def variable_names(self, variables):
        return ["x_{}".format(i) for i in variables]

    def matches_constraints(self, row):
        pass

    def bounds(self, i, d):
        pass

    def draw2d(self, df, selected=[0, 1]):
        cols = df.columns
        df['valid'] = df.apply(self.matches_constraints, axis=1)
        color = np.where(df['valid'].values == True, 'g', 'r')
        df.plot(kind='scatter', x=cols[selected[0]], y=cols[selected[1]], c=color)
        plt.show()

    def info(self, df):
        print(df.describe())
        print(df)

    def generate_df(self):
        cols = self.variable_names(self.variables)
        samples = sampler.samples(self.bounds, rows=BenchmarkModel.rows, cols=len(self.variables))
        return pd.DataFrame(samples.T, columns=cols)