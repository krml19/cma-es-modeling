import numpy as np
import pandas as pd

from scripts.benchmarks.ball import Ball
from scripts.benchmarks.cube import Cube
from scripts.benchmarks.simplex import Simplex
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.csv.file_helper import Paths


class DataModel:
    def __init__(self, name, B, n):

        self.benchmark_model = self.__get_model(name=name, B=B, n=n)

    def __get_model(self, name, B, n) -> BenchmarkModel:
        return {
            'ball': Ball(i=n, B=B),
            'cube': Cube(i=n, B=B),
            'simplex': Simplex(i=n, B=B),
        }[name]

    def __filename(self, path) -> str:
        return "{}{}.csv".format(path, self.benchmark_model.filename())

    def __get_dataset(self, filename):
        return pd.read_csv(filename).values

    def train_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.train.value))

    def test_set(self) -> tuple:
        test = self.__get_dataset(self.__filename(Paths.test.value))
        return test[:, :-1], test[:, -1]

    def valid_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.valid.value))

