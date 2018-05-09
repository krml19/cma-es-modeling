import numpy as np

from scripts.utils.sampler import Sampler
import pandas as pd
from scripts.csv import file_helper as fh
from scripts.utils.logger import Logger


class BenchmarkModel:

    def __init__(self, i, d=2.7, train_rows=5000, test_rows=int(1e5), L=1e10, name='base-model', B=list([1, 1]), seed=404):
        self.variables = np.arange(1, i + 1)
        self.i = i
        self.d = d * np.ones(i)
        self.name = name
        self.train_rows = train_rows
        self.test_rows = test_rows
        self.L = L
        self.B = B
        self.k = len(self.B)
        self.seed = seed
        self.logger = Logger(name=name)
        self.bounds = None
        self.sampler = Sampler(seed=seed)

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

    def generate_train_dataset(self):
        df = self.__generate_train_subset()

        while df.shape[0] < self.train_rows:
            df = df.append(self.__generate_train_subset())

        self.logger.debug('Removing invalid points and validation column')
        df = df.drop(['valid'], axis=1, errors='ignore')

        self.__save(df=df.head(self.train_rows), path=fh.Paths.train.value)
        return df

    def __generate_train_subset(self):
        cols = self.variable_names(self.variables)
        self.logger.debug('Sampling points')
        samples = self.sampler.samples(self.bounds, rows=self.train_rows, cols=len(self.variables))
        self.logger.debug('Creating data frame')
        df = pd.DataFrame(samples.T, columns=cols)
        self.logger.debug('Generating validation columns')
        df['valid'] = self.generate_valid_column(df)
        return df[(df['valid'] == True)].reset_index(drop=True)

    def __save(self, df, path):
        filename = self.filename()
        self.logger.debug('Saving: {}{}'.format(path, filename))
        fh.write_data_frame(df=df, filename=filename, path=path)
        self.logger.debug('Saved: {}{}'.format(path, filename))

    def generate_validation_dataset(self):
        cols = self.variable_names(self.variables)
        samples = self.sampler.samples(self.bounds, rows=self.test_rows, cols=len(self.variables))
        df = pd.DataFrame(samples.T, columns=cols)

        self.__save(df=df, path=fh.Paths.valid.value)
        return df

    def generate_test_dataset(self):
        cols = self.variable_names(self.variables)
        samples = self.sampler.samples(self.bounds, rows=self.test_rows, cols=len(self.variables))
        df = pd.DataFrame(samples.T, columns=cols)
        self.logger.debug('Generating validation columns')
        df['valid'] = self.generate_valid_column(df)
        self.__save(df=df, path=fh.Paths.test.value)
        return df

    def generate_datasets(self):
        self.generate_train_dataset()
        self.generate_validation_dataset()
        self.generate_test_dataset()

    def filename(self) -> str:
        return "{}_{}_{}".format(self.name, self.i, self.k)

    def benchmark_objective_function(self, X: np.ndarray, w: np.ndarray, w0: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda x: self.matches_constraints(x), 1, X)
