import numpy as np
import pandas as pd

from scripts.benchmarks.ball import Ball
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.csv.file_helper import Paths


class DataModel:

    def __init__(self, model: BenchmarkModel = Ball):
        self.benchmark_model = model

    def __filename(self, path) -> str:
        return "{}{}{}_seed_{}_k_{}.csv".format(path, self.benchmark_model.name, self.benchmark_model.i, self.benchmark_model.seed, self.benchmark_model.k)

    def __get_dataset(self, filename):
        return pd.read_csv(filename).values

    def train_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.train.value))

    def test_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.test.value))

    def valid_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.valid.value))

