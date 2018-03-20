import numpy as np
import matplotlib.pyplot as plt


class BenchmarkModel:
    rows = 1000

    def __init__(self, i, d=2.7):
        self.variables = np.arange(1, i + 1)
        self.d = d * np.ones(i)

    def variable_names(self, variables):
        return ["x_{}".format(i) for i in variables]

    def matches_constraints(self, row):
        pass

    def bounds(self, i, d):
        pass

    def draw2d(self, df, cols, selected=[0, 1]):
        df['valid'] = df.apply(self.matches_constraints, axis=1)
        color = np.where(df['valid'].values == True, 'g', 'r')
        df.plot(kind='scatter', x=cols[selected[0]], y=cols[selected[1]], c=color)
        plt.show()

    def info(self, df):
        print(df.describe())
        print(df)