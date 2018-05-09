import numpy as np
import itertools
import random


def samples(bounds, rows, cols=1):
    np.random.seed(seed=np.random.randint(int(1e4)))
    return np.matrix(
        [np.random.uniform(low=low, high=high, size=(rows, cols)).flatten() for low, high in bounds])


def ct(arr, r):
    a = np.concatenate((np.array([2 * np.pi]), arr))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return np.round(si * co * r, decimals=5)


def cartesian(n: int, dim: int, r: float=1):
    assert dim >= 2
    assert n >= dim

    _n = np.ceil(n/(dim-1))
    phis = np.arange(_n) / _n * 2 * np.pi

    phis = [np.array(i) for i in itertools.product(phis, repeat=dim-1)]
    random.shuffle(phis)
    phis = itertools.islice(phis, n)

    coordinates = [ct(phi, r=r) for phi in phis]
    return np.concatenate(coordinates)


def scale_factor(train_data_set: np.array, margin: float):
    return np.amax(train_data_set, axis=0) * margin


def bounding_sphere(n: int, train_data_set: np.array, dim: int, r=1, margin: float=2.0):
    x0 = cartesian(n, r=r, dim=dim)
    dividers = np.tile(scale_factor(train_data_set=train_data_set, margin=margin), n)
    x0 = x0 / dividers
    return x0

