import numpy as np
import itertools
import random
import math

def samples(bounds, rows):
    np.random.seed(seed=np.random.randint(int(1e4)))
    return np.matrix(
        [np.random.uniform(low=low, high=high, size=(rows, 1)).flatten() for low, high in bounds])


def ct(arr, r):
    a = np.concatenate((np.array([2 * np.pi]), arr))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return np.round(si * co * r, decimals=5)


def min_pow(x: int, value: int):
    p = x ** x
    return x if p >= value else min_pow(x=x+1, value=value)


def cartesian(n: int, dim: int, r: float=1):
    assert dim >= 2
    assert n >= dim

    _n = min_pow(2, n)
    _n = _n if dim > 2 else _n * 2
    phis = np.arange(_n) / _n * 2 * np.pi

    phis = [np.array(i) for i in itertools.product(phis, repeat=int(dim - 1))]
    random.shuffle(phis)
    phis = itertools.islice(phis, 0, int(n))

    coordinates = [ct(phi, r=r) for phi in phis]
    return np.concatenate(coordinates)


def scale_factor(train_data_set: np.array, margin: float):
    return np.amax(train_data_set, axis=0) * margin


def bounding_sphere(n: int, train_data_set: np.array, dim: int, r=1, margin: float=2.0):
    x0 = cartesian(n, r=r, dim=dim)
    dividers = np.tile(scale_factor(train_data_set=train_data_set, margin=margin), n)
    x0 = x0 / dividers
    return x0

