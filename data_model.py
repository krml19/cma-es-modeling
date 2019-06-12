import re

import numpy as np
import pandas as pd

from models import BenchmarkModel, Cube, Simplex, Ball
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
            'ball': Ball(n=n, B=B),
            'cube': Cube(n=n, B=B),
            'simplex': Simplex(n=n, B=B)
        }[name]

    def __problem(self, name):
        return {
            'ball': Problem.Ball,
            'cube': Problem.Cube,
            'simplex': Problem.Simplex,
            'case_study': None
        }[name]

    def __filename(self, path) -> str:
        return "{}_{}.csv.xz".format(path, self.filename)

    def __get_dataset(self, filename, nrows=None):
        return pd.read_csv(filename, nrows=nrows).values

    def train_set(self) -> np.ndarray:
        return self.__get_dataset(self.__filename(Paths.train.value), nrows=self.train_sample)

    def dataset(self, seed):
        # while True:
        #     test = sample.dataset(int(1e6), n=self.n, k=self.k, seed=seed, p=self.problem, classes=[0, 1])
        #     _sum = test[:, -1].sum().astype(int)
        #     if _sum > 3:
        #         return test
        #     seed = seed + 1

        test = sample.dataset(int(1e6), n=self.n, k=self.k, seed=seed, p=self.problem, classes=[0, 1])
        return test

    def test_set(self) -> tuple:
        test = self.dataset(seed=self.seed + 100)
        return test[:, :-1].astype(float), test[:, -1].astype(int)

    def valid_set(self) -> np.ndarray:
        valid = self.dataset(seed=self.seed + 1000)
        return valid[:, :-1].astype(float)

    def valid_set2(self) -> np.ndarray:
        valid = self.dataset(seed=self.seed + 10000)
        return valid[:, :-1].astype(float)

class CaseStudyBenchmarkModel:
    def __init__(self):
        self.n = 8
        self.k = 1
        self.d = 0
        self.name = 'case_study'

class CaseStudyDataModel:
    def __init__(self):
        self.benchmark_model = CaseStudyBenchmarkModel()
        self.train = pd.read_csv('./datasets/Rice-nonoise2.csv', delimiter=';')
        self.train.drop(['Type'], inplace=True, axis=1)
        self.n = len(self.train.keys())

        self.domains = [(float(re.sub(r'.*\|(.*)\|.*', r'\1', key)), float(re.sub(r'.*\|(.*)\].*', r'\1', key))) for key in self.train.keys()]
        self.valid1 = self.__sample_valid()
        self.valid2 = self.__sample_valid()

    def __sample_valid(self, count=1000000):
        n = self.n

        domains = np.array(self.domains, np.float64)
        assert domains.shape[0] == n
        assert domains.shape[1] == 2

        min_ = domains[:, 0]
        max_ = domains[:, 1]
        maxmin_ = max_ - min_
        assert (min_ <= max_).all()

        random_matrix = np.random.sample((count, n))
        np.multiply(random_matrix, maxmin_, out=random_matrix)
        np.add(random_matrix, min_, out=random_matrix)
        assert (min_ <= random_matrix).all() and (random_matrix <= max_).all()

        return random_matrix

    def train_set(self) -> np.ndarray:
        return self.train

    def test_set(self) -> tuple:
        return self.train, np.ones((self.train.shape[0], 1))

    def valid_set(self) -> np.ndarray:
        return self.valid1

    def valid_set2(self) -> np.ndarray:
        return self.valid2
