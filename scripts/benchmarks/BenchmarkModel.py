import numpy as np


class BenchmarkModel:

    def __init__(self, i, d=2.7):
        self.variables = np.arange(1, i + 1)
        self.d = d * np.ones(i)

    def variable_names(self, variables):
        return ["x_{}".format(i) for i in variables]