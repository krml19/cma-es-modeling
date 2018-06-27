import numpy as np
import pandas as pd

from ball import Ball
from cube import Cube
from simplex import Simplex
from benchmark_model import BenchmarkModel
from file_helper import Paths
import Problem
import sample
import sampler

class DataModel:
    def __init__(self, name, k, n, seed, train_sample=500):

        self.benchmark_model = self.__get_model(name=name, k=k, n=n)
        self.filename = '{}_{}_{}_{}'.format(name.title(), k, n, seed)
        self.problem = self.__problem(name)
        self.seed = seed
        self.n = n
        self.k = k
        self.train_sample = train_sample

    def __get_model(self, name, k, n) -> BenchmarkModel:
        B = [1] * k
        return {
            'ball': Ball(i=n, B=B),
            'cube': Cube(i=n, B=B),
            'simplex': Simplex(i=n, B=B),
        }[name]

    def __problem(self, name):
        return {
            'ball': Problem.Ball,
            'cube': Problem.Cube,
            'simplex': Problem.Simplex
        }[name]

    def __filename(self, path) -> str:
        return "{}_{}.csv.xz".format(path, self.filename)

    def __get_dataset(self, filename, nrows=None):
        return pd.read_csv(filename, nrows=nrows).values

    def train_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.train.value), nrows=self.train_sample)

    def dataset(self, seed):
        while True:
            test = sample.dataset(int(1e6), n=self.n, k=self.k, seed=seed, p=self.problem, classes=[0, 1])
            _sum = test[:, -1].sum().astype(int)
            if _sum > 3:
                return test
            seed = seed + 1

    def test_set(self) -> tuple:
        test = self.dataset(seed=self.seed + 100)
        return test[:, :-1].astype(float), test[:, -1].astype(int)

    def valid_set(self) -> np.ndarray:
        valid = self.dataset(seed=self.seed + 1000)
        return valid[:, :-1].astype(float)

    def valid_set2(self) -> np.ndarray:
        valid = self.dataset(seed=self.seed + 10000)
        return valid[:, :-1].astype(float)
