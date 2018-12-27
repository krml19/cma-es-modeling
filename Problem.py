import math
import numpy as np
import pandas as pd
import os.path


class Problem:

    def __init__(self, name: str, data: np.ndarray = None, target: np.ndarray = None):
        self.__name: str = name
        self.__data = data
        self.__target = target

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def data(self) -> np.ndarray:
        return self.__data

    @data.setter
    def data(self, data: np.ndarray):
        self.__data = data

    @property
    def target(self) -> np.ndarray:
        return self.__target

    @target.setter
    def target(self, target: np.ndarray):
        self.__target = target

    @staticmethod
    def from_csv(filename: str, size_limit=None):
        frame = pd.read_csv(filename, nrows=size_limit)
        if 'y' in frame.axes[1]:
            y_index = list(frame.axes[1]).index('y')
            y = frame.values[:, y_index]
            X = np.concatenate((frame.values[:, :y_index], frame.values[:, y_index + 1:]), axis=1)
        else:
            y = None
            X = frame.values
        return Problem(os.path.basename(filename), X, y)


class Benchmark:
    def __init__(self, name: str, k: int, n: int, d=2.7):
        self.__name: str = name
        self.__k: int = k
        self.__n: int = n
        self.__d: float = d
        self.__W: np.ndarray = None
        self.__b: np.ndarray = None
        self.__delimiters: list = None
        self.__domains: list = None

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def k(self) -> int:
        return self.__k

    @property
    def n(self) -> int:
        return self.__n

    @property
    def d(self) -> float:
        return self.__d

    @property
    def W(self) -> np.ndarray:
        return self.__W

    @W.setter
    def W(self, W: np.ndarray):
        self.__W = W

    @property
    def b(self) -> np.ndarray:
        return self.__b

    @b.setter
    def b(self, b: np.ndarray):
        self.__b = b

    @property
    def delimiters(self) -> list:
        return self.__delimiters

    @delimiters.setter
    def delimiters(self, delimiters: list):
        self.__delimiters = delimiters

    @property
    def domains(self) -> list:
        return self.__domains

    @domains.setter
    def domains(self, domains: list):
        self.__domains = domains

    def predict(self, X: np.ndarray) -> np.ndarray:
        delimiters = self.delimiters

        WXb = np.matmul(self.W, X.T).T <= self.b
        cls = np.zeros(X.shape[0], np.bool_)
        for d in range(len(delimiters) - 1):
            cls_d = WXb.T[delimiters[d]:delimiters[d + 1]].all(axis=0)
            assert cls_d.shape[0] == cls.shape[0]
            np.add(cls, cls_d, out=cls)
        assert cls.shape[0] == X.shape[0]
        return cls

    def as_constraints_text(self, variable_names=[]) -> list:
        c = []
        for d in range(len(self.delimiters) - 1):
            for i, w in enumerate(self.W[self.delimiters[d]:self.delimiters[d + 1]]):
                constr = " + ".join(["%f%s" % (wl, variable_names[l] if l < len(variable_names) else "x%d" % (l + 1)) for l, wl in enumerate(w) if abs(wl) > 1e-6]) + " + Mb%d <= %f + M" % (
                    d + 1, self.b[i])
                c.append(constr)

        c.append(" + ".join(["b%d" % d for d in range(1, len(self.delimiters))]))

        assert len(c) == self.W.shape[0] + 1, self.W.shape
        return c

    def sample(self, count: int, allowed_classes=[1]) -> Problem:
        '''Samples examples'''
        n = self.n

        domains = np.array(self.domains, np.float64)
        assert domains.shape[0] == n
        assert domains.shape[1] == 2

        min_ = domains[:, 0]
        max_ = domains[:, 1]
        maxmin_ = max_ - min_
        assert (min_ <= max_).all()

        X = np.empty((count, n), np.float64)
        y = np.empty(count, np.float64)
        l = 0
        while l < count:
            random_matrix = np.random.sample((count, n))
            np.multiply(random_matrix, maxmin_, out=random_matrix)
            np.add(random_matrix, min_, out=random_matrix)
            assert (min_ <= random_matrix).all() and (random_matrix <= max_).all()

            y_pred = self.predict(random_matrix)
            with_allowed_class = np.in1d(y_pred, allowed_classes)
            assert with_allowed_class.dtype == np.bool_

            X_valid = random_matrix[with_allowed_class]
            y_valid = y_pred[with_allowed_class]
            if X_valid.shape[0] > 0:
                X[l:min(l + X_valid.shape[0], count)] = X_valid[:min(X_valid.shape[0], count - l)]
                y[l:min(l + X_valid.shape[0], count)] = y_valid[:min(X_valid.shape[0], count - l)]
                l += X_valid.shape[0]

        assert np.in1d(self.predict(X), allowed_classes).all()
        return Problem(self.name, X, y)


class Cube(Benchmark):
    def __init__(self, k: int, n: int):
        super().__init__("Cube", k, n)
        self.W = np.zeros((2 * n * k, n), dtype=np.float32)
        self.b = np.empty((2 * n * k), dtype=np.float32)

        for j in range(k):
            for i in range(n):
                index = 2 * j * n + 2 * i
                self.W[index, i] = -1.0
                self.W[index + 1, i] = 1.0
                self.b[index] = -(i + 1) * (j + 1)
                self.b[index + 1] = (i + 1) * (j + 1) + (i + 1) * self.d

        self.delimiters = list(range(0, 2 * k * n + 1, 2 * n))
        self.domains = [((i + 1) - (i + 1) * k * self.d, (i + 1) + 2 * (i + 1) * k * self.d) for i in range(n)]

        assert self.delimiters[-1:][0] == self.W.shape[0], self.W.shape
        assert self.delimiters[-1:][0] == self.b.shape[0]


class Simplex(Benchmark):
    def __init__(self, k: int, n: int):
        super().__init__("Simplex", k, n)
        self.W = np.zeros((k * n * (n - 1) + k, n), dtype=np.float32)
        self.b = np.empty((k * n * (n - 1) + k), dtype=np.float32)
        self.delimiters = [0]
        self.domains = [(-1.0, 2 * k + self.d) for _ in range(n)]

        tanpi12 = math.tan(math.pi / 12)
        cotpi12 = 1 / tanpi12

        index = 0
        for j in range(k):
            for i in range(n):
                for l in range(i + 1, n):
                    self.W[index, i] = -cotpi12
                    self.W[index, l] = tanpi12
                    self.b[index] = -2 * j
                    index += 1
                    self.W[index, l] = -cotpi12
                    self.W[index, i] = tanpi12
                    self.b[index] = -2 * j
                    index += 1

            self.W[index] = np.ones(n, dtype=np.float32)
            self.b[index] = (j + 1) * self.d
            index += 1
            self.delimiters.append(index)

        assert index == self.W.shape[0]
        assert self.delimiters[-1:][0] == self.W.shape[0], self.delimiters
        assert self.delimiters[-1:][0] == self.b.shape[0]


class Ball(Benchmark):
    def __init__(self, k: int, n: int):
        super().__init__("Ball", k, n)
        self.twosqrt6d = 2 * math.sqrt(6) * self.d
        self.d2 = self.d * self.d
        self.domains = [((i + 1) - 2 * self.d, (i + 1) + 2 * math.sqrt(6) * (k - 1) * self.d / math.pi + 2 * self.d) for i in range(n)]

    def predict(self, X: np.ndarray) -> np.ndarray:

        y = np.zeros(X.shape[0], dtype=np.int8)
        for l, x in enumerate(X):
            for j in range(self.k):
                if sum((xi - i - 1 - self.twosqrt6d * j / ((i + 1) * math.pi)) ** 2 for i, xi in enumerate(x)) <= self.d2:
                    y[l] = 1
                    break
        return y

    # def sample(self, count: int, allowed_classes: list = [1]):
    #     '''Samples examples'''
    #     domains = np.array(self.domains, np.float32)
    #     assert domains.shape[0] == self.n
    #     assert domains.shape[1] == 2
    #
    #     min_ = domains[:, 0]
    #     max_ = domains[:, 1]
    #     maxmin_ = max_ - min_
    #     assert (min_ <= max_).all()
    #
    #     X = np.empty((count, self.n), np.float32)
    #     for l in range(count):
    #         while True:
    #             sample = np.random.random_sample(self.n) * maxmin_ + min_
    #             y = any(sum((xi - i - 1 - self.twosqrt6d * j / ((i + 1) * math.pi)) ** 2 for i, xi in enumerate(sample)) <= self.d2 for j in range(self.k))
    #             if y in allowed_classes:
    #                 break
    #         X[l] = sample
    #
    #     assert np.in1d(self.predict(X), allowed_classes).all()
    #     return X
