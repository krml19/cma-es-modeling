import numpy as np
from scripts.utils import sampler
import pandas as pd
from scripts.csv import file_helper as fh
from scripts.utils.logger import Logger


class BenchmarkModel:

    def __init__(self, i, d=2.7, rows=1000, L=1e10, name='base-model', B=list([1]), seed=404):
        self.variables = np.arange(1, i + 1)
        self.i = i
        self.d = d * np.ones(i)
        self.name = name
        self.rows = rows
        self.L = L
        self.B = B
        self.k = len(self.B)
        self.seed = seed
        self.logger = Logger(name=name)

    def variable_names(self, variables):
        return ["x_{}".format(i) for i in variables]

    def matches_constraints(self, row):
        pass

    def _bounds(self, i, d):
        pass

    def generate_valid_column(self, df):
        return df.apply(self.matches_constraints, axis=1)

    def info(self, df):
        print(df.describe())
        print(df)

    def generate_df(self, take_only_valid_points=True):
        cols = self.variable_names(self.variables)
        self.logger.debug('Sampling points')
        samples = sampler.samples(self.bounds, rows=self.rows, cols=len(self.variables), seed=self.seed)
        self.logger.debug('Creating data frame')
        df = pd.DataFrame(samples.T, columns=cols)
        self.logger.debug('Generating validation columns')
        df['valid'] = self.generate_valid_column(df)
        if take_only_valid_points:
            df = df[df.valid == True]
            self.logger.debug('Removing invalid points and validation column')
            df = df.drop(['valid'], axis=1, errors='ignore')
        return df

    def save(self, df, path='data/train/'):
        filename = "{}{}_seed_{}".format(self.name, len(self.variables), self.seed)
        self.logger.debug('Saving: {}{}'.format(path, filename))
        fh.write_data_frame(df=df, filename=filename, path=path)
        self.logger.debug('Saved: {}{}'.format(path, filename))

    def generate_validation_dataset(self):
        cols = self.variable_names(self.variables)
        samples = sampler.samples(self.bounds, rows=self.rows, cols=len(self.variables))
        df = pd.DataFrame(samples.T, columns=cols)

        fh.write_validation_file(df=df, filename="{}{}_seed_{}".format(self.name, len(self.variables), self.seed))

    def bounding_sphere(self):
        w = list()
        for i, d in enumerate(self.d, start=1):
            _bounds = self._bounds(i=i, d=d)
            for j, bound in enumerate(_bounds):
                wi = np.zeros(self.i)
                wi[j] = -1 / bound
                w.append(wi)
        return np.concatenate(w)
