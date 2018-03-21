import numpy as np
import matplotlib.pyplot as plt
from scripts import sampler
import pandas as pd


class BenchmarkModel:
    rows = 100000

    def __init__(self, i, d=2.7):
        self.variables = np.arange(1, i + 1)
        self.d = d * np.ones(i)

    def variable_names(self, variables):
        return ["x_{}".format(i) for i in variables]

    def matches_constraints(self, row):
        pass

    def bounds(self, i, d):
        pass

    def generate_valid_column(self, df):
        return df.apply(self.matches_constraints, axis=1)

    def info(self, df):
        print(df.describe())
        print(df)

    def generate_df(self, take_only_valid_points=True):
        cols = self.variable_names(self.variables)
        samples = sampler.samples(self.bounds, rows=BenchmarkModel.rows, cols=len(self.variables))
        df = pd.DataFrame(samples.T, columns=cols)
        df['valid'] = self.generate_valid_column(df)
        if take_only_valid_points:
            df = df[df.valid == True]
        return df
