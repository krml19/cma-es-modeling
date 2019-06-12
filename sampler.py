import numpy as np
import itertools
import random
import math


def samples(bounds: list, rows: int, seed: int):
    np.random.seed(seed=seed)
    return np.vstack(
        [np.random.uniform(low=low, high=high, size=rows) for low, high in bounds]).T


# Source: https://stackoverflow.com/a/20133681
def ct(arr, r):
    a = np.concatenate((np.array([2 * np.pi]), arr))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return np.round(si * co * r, decimals=5)


def min_pow(n, x: int, value: int):
    p = np.power(x, n-1)
    return x if p > value else min_pow(n=n, x=x+1, value=value)


def cartesian(n: int, dim: int, r: float=1):
    assert dim >= 2
    assert n >= dim

    _n = min_pow(dim, 2, n)
    _n = _n if dim > 2 else _n * 2
    phis = (np.arange(_n) + 0.5) / _n * 2 * np.pi

    phis = [np.array(i) for i in itertools.product(phis, repeat=int(dim - 1))]
    coordinates = [ct(phi, r=r) for phi in phis]

    unique_coordinates = np.unique(coordinates, axis=0)
    np.random.shuffle(unique_coordinates)

    if unique_coordinates.shape[0] > n:
        unique_coordinates = unique_coordinates[0:n, :]
    while unique_coordinates.shape[0] < n:
        unique_coordinates = np.vstack((unique_coordinates, unique_coordinates[0:n-unique_coordinates.shape[0], :]))

    return unique_coordinates



def bounding_sphere(n: int, train_data_set: np.array, dim: int, r=1, margin: float=2.0):
    # sign = np.vectorize(lambda x: 1 if x >= 0 else -1)
    W = np.array(cartesian(n, r=r, dim=dim))  # matrix of constraint * weights in constraints
    XW = np.matmul(train_data_set, W.T)  # matrix of example * constraint (example in row, constraint rhs in column)
    a = np.amax(XW, axis=0)  # vector of maximal rhs per constraint
    Wnorm = W / np.tile(a, (dim, 1)).T  # normalize constraints such that rhs=1
    Wnorm /= margin  # expand/shrink margin, rhs=1/margin
    assert np.all(np.matmul(train_data_set, Wnorm.T) <= 1.0/margin + 1e-6)
    return Wnorm.flatten()

