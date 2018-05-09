import numpy as np
import pandas as pd

from scripts.benchmarks.ball import Ball
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.csv.file_helper import Paths


class DataModel:

    def __init__(self, model: BenchmarkModel = Ball):
        self.benchmark_model = model

    def __filename(self, path) -> str:
        return "{}{}.csv".format(path, self.benchmark_model.filename())

    def __get_dataset(self, filename):
        #  FIXME: Remove `nrows` parameter
        return pd.read_csv(filename, nrows=100).values

    def train_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.train.value))

    def test_set(self) -> tuple:
        test = self.__get_dataset(self.__filename(Paths.test.value))
        return test[:, :-1], test[:, -1]

    def valid_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.valid.value))

