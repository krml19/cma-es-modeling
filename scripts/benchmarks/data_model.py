import numpy as np
import pandas as pd

from scripts.benchmarks.ball import Ball
from scripts.benchmarks.cube import Cube
from scripts.benchmarks.simplex import Simplex
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.csv.file_helper import Paths


class DataModel:
    def __init__(self, name, k, n, seed):

        self.benchmark_model = self.__get_model(name=name, k=k, n=n)
        self.filename = '{}_{}_{}_{}'.format(name.title(), k, n, seed)

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
        return self.__get_dataset(self.__filename(Paths.train.value), nrows=500)

    def test_set(self) -> tuple:
        test = self.__get_dataset(self.__filename(Paths.test.value))
        return test[:, :-1].astype(float), test[:, -1].astype(int)

    def valid_set(self) -> np.ndarray:
        valid = self.__get_dataset(self.__filename(Paths.valid.value))
        return valid[:, :-1].astype(float)

