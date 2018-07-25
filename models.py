import numpy as np
import pandas as pd

import sampler
from constraint import Constraints, Constraint, Operator
from file_helper import Paths
from logger import Logger

class BenchmarkModel:

    def __init__(self, n, d=2.7, L=1e10, name='base-model', B=list([1, 1])):
        self.variables = np.arange(1, n + 1)
        self.n = n
        self.d = d
        self.name = name
        self.L = L
        self.B = B
        self.k = len(self.B)
        self.logger = Logger(name=name)
        self.bounds = None

    def variable_names(self, variables):
        return ["x_{}".format(i) for i in variables]

    def matches_constraints(self, row):
        pass

    def _bounds(self, i, d):
        pass

    def valid(self, X: np.ndarray):
        return np.apply_along_axis(self.matches_constraints, 1, X).astype(int)

    def generate_valid_column(self, df):
        return df.apply(self.matches_constraints, axis=1)

    def generate_train_dataset(self, rows, seed):
        df = self.__generate_train_subset(seed=seed)

        while df.shape[0] < rows:
            df = df.append(self.__generate_train_subset())
            self.logger.debug("Current progress: {}/{}".format(len(df), rows))

        self.logger.debug('Removing invalid points and validation column')
        df = df.drop(['valid'], axis=1, errors='ignore')
        return df.head(rows)

    def __generate_train_subset(self, seed=None, sample_size=int(1e6)):
        cols = self.variable_names(self.variables)
        self.logger.debug('Sampling points')
        samples = sampler.samples(self.bounds, rows=sample_size, seed=seed)
        self.logger.debug('Creating data frame')
        df = pd.DataFrame(samples, columns=cols)
        self.logger.debug('Generating validation columns')
        df['valid'] = self.generate_valid_column(df)
        return df[(df['valid'] == True)].reset_index(drop=True)

    def dataset(self, seed, labeled: bool = False, nrows = int(1e6)):
        cols = self.variable_names(self.variables)
        samples = sampler.samples(self.bounds, rows=nrows, seed=seed)
        df = pd.DataFrame(samples, columns=cols)
        if labeled:
            df['valid'] = self.generate_valid_column(df)
        assert df.shape[0] == nrows
        return df.values

    def benchmark_objective_function(self, X: np.ndarray, w: np.ndarray, w0: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda x: self.matches_constraints(x), 1, X)


class Ball(BenchmarkModel):

    def _bounds(self, i, d):
        return i - 2 * d, i + 2 * d + (2 * np.sqrt(6) * (self.k - 1) * d) / np.pi

    def __init__(self, n, d=2.7, B=list([1, 1])):
        super().__init__(n=n, d=d, name='ball', B=B)

        self.constraint_sets = [
            Constraints(constraints=[Constraint(_operator=Operator.gt, value=d * d + self.L * (1 - bj))]) for bj in
            self.B]

        self.bounds = [self._bounds(i=ii, d=d) for ii in self.variables]

    def matches_constraints(self, row):
        row = row.values if hasattr(row, 'values') else row
        validation_result = [self._match_constraint_set(constraint=constraints, j=j, row=row) for j, constraints in
                             enumerate(self.constraint_sets)]
        return sum(validation_result) >= 1

    def _match_constraint_set(self, constraint, j, row):
        validation_result = [(x - i - (2 * np.sqrt(6) * j * self.d) / (i * np.pi)) ** 2 for i, x in enumerate(row, start=1)]
        _sum = sum(validation_result)
        return constraint.validate(_sum)


class Simplex(BenchmarkModel):

    def _bounds(self, i, d):
        return -1, 2 * self.k + d

    def __init__(self, n, d=2.7, B=list([1, 1])):
        super().__init__(n=n, d=d, name='simplex', B=B)

        self.constraint_sets = [
            Constraints(constraints=[Constraint(_operator=Operator.lt, value=2 * j - 2 - self.L * (1 - bj)),
                                     Constraint(_operator=Operator.lt, value=2 * j - 2 - self.L * (1 - bj)),
                                     Constraint(_operator=Operator.gt, value=j * d + self.L * (1 - bj))]) for j, bj in
            enumerate(self.B, start=1)]

        self.bounds = [self._bounds(i=ii, d=d) for ii in self.variables]

    def matches_constraints(self, row):
        validation_result = [self._matches_constraints_set(constraints=constraints, row=row) for constraints in
                             self.constraint_sets]
        return sum(validation_result) >= 1

    def _matches_constraints_set(self, constraints, row):
        _sum = sum(row)
        if not constraints.constraints[2].match(_sum):
            return False
        values = row.values if hasattr(row, 'values') else row

        for xi, xj in zip(values[::1], values[1::1]):
            if not constraints.constraints[0].match(Simplex.constraint1(xi, xj)):
                return False
            if not constraints.constraints[1].match(Simplex.constraint2(xi, xj)):
                return False
        return True

    @staticmethod
    def tan_pi_12(x):
        return x * np.tan(np.pi / 12)

    @staticmethod
    def cot_pi_12(x):
        return x / (np.tan(np.pi / 12))

    @staticmethod
    def constraint1(xi, xj):
        return Simplex.cot_pi_12(xi) - Simplex.tan_pi_12(xj)

    @staticmethod
    def constraint2(xi, xj):
        return Simplex.cot_pi_12(xj) - Simplex.tan_pi_12(xi)


class Cube(BenchmarkModel):

    def _bounds(self, i, d):
        return i - i * d * self.k, i + 2 * i * d * self.k

    def __init__(self, n, d=2.7, B=list([1, 1])):
        super().__init__(n=n, d=d, name='cube', B=B)
        self.constraint_sets = list()
        for j, bj in enumerate(self.B, start=1):
            self.constraint_sets.append([
                Constraints(constraints=[Constraint(_operator=Operator.lt, value=it * j - self.L + self.L * bj),
                                         Constraint(_operator=Operator.gt,
                                                    value=it * j + it * d + self.L - self.L * bj)])
                for it in self.variables])

        self.bounds = [self._bounds(i=ii, d=d) for ii in self.variables]

    def matches_constraints(self, row):
        validation_result = [self.match(constraints, row) for constraints in self.constraint_sets]
        return sum(validation_result) >= 1

    def match(self, constraints, row):
        for index, xi in enumerate(row):
            if not constraints[index].validate(xi):
                return False
        return True

    def constraints(self):
        for i, constraints in enumerate(self.constraint_sets, start=1):
            print("k=%d" % i)
            for constraint in constraints:
                print(constraint)


class DataModel:
    def __init__(self, name, k, n, seed, train_sample=500):

        self.benchmark_model = self.__get_model(name=name, k=k, n=n)
        self.filename = '{}_{}_{}_{}'.format(name.title(), k, n, seed)
        self.seed = seed
        self.n = n
        self.k = k
        self.train_sample = train_sample

    def __get_model(self, name, k, n) -> BenchmarkModel:
        B = [1] * k
        return {
            'ball': Ball(n=n, B=B),
            'cube': Cube(n=n, B=B),
            'simplex': Simplex(n=n, B=B),
        }[name]

    def __filename(self, path) -> str:
        return "{}/training_{}.csv.xz".format(path, self.filename)

    def __get_dataset(self, filename, nrows=None):
        return pd.read_csv(filename, nrows=nrows).values

    def train_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.train.value), nrows=self.train_sample)

    def test_set(self) -> tuple:
        test = self.benchmark_model.dataset(seed=self.seed + 100, labeled=True)
        return test[:, :-1].astype(float), test[:, -1].astype(int)

    def valid_set(self) -> np.ndarray:

        valid = self.benchmark_model.dataset(seed=self.seed + 1000)
        return valid.astype(float)

    def valid_set2(self) -> np.ndarray:
        valid = self.benchmark_model.dataset(seed=self.seed + 10000)
        return valid.astype(float)