import numpy as np
import pandas as pd
import sampler

from ball import Ball
from cube import Cube
from simplex import Simplex
from benchmark_model import BenchmarkModel
from file_helper import Paths


class DataModel:
    test_n_rows = int(1e6)

    def __init__(self, name, k, n, seed, train_samples: int = 500):

        self.benchmark_model = self.__get_model(name=name, k=k, n=n)
        self.filename = '{}_{}_{}_{}'.format(name.title(), k, n, seed)
        self.seed = seed
        self.n = n
        self.k = k
        self.train_samples = train_samples

    def __get_model(self, name, k, n) -> BenchmarkModel:
        B = [1] * k
        return {
            'ball': Ball(i=n, B=B),
            'cube': Cube(i=n, B=B),
            'simplex': Simplex(i=n, B=B),
        }[name]

    def __filename(self, path) -> str:
        return "{}_{}.csv.xz".format(path, self.filename)

    def __get_dataset(self, filename, nrows=None):
        return pd.read_csv(filename, nrows=nrows).values

    def train_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.train.value), nrows=self.train_samples)

    def data_set(self, seed: int):
        while True:
            samples = sampler.samples(self.benchmark_model.bounds, rows=DataModel.test_n_rows, seed=seed)
            if samples[:, -1].astype(int).sum() > 3:
                break
            seed = seed + 1
        return samples

    def test_set(self) -> tuple:
        seed = self.seed + 100
        test = self.data_set(seed=seed)

        return test.astype(float), self.benchmark_model.valid(test)

    def valid_set(self) -> np.ndarray:
        seed = self.seed + 1000
        valid = self.data_set(seed=seed)
        return valid.astype(float)
