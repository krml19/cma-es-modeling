import numpy as np
import itertools
import random


class Sampler:

    def __init__(self, seed: int = 404):
        np.random.seed(seed)

    def samples(self, bounds, rows, cols=1):
        return np.matrix(
            [np.random.uniform(low=low, high=high, size=(rows, cols)).flatten() for low, high in bounds])

    def uniform_samples(self, low, high, size):
        return np.random.uniform(low=low, high=high, size=size)


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


# def points_from_n_sphere(dim=3, N=600):
#     norm = np.random.normal
#     normal_deviates = norm(size=(dim, N))
#
#     radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
#     points = normal_deviates / radius
#     return points


def scale_factor(train_data_set: np.array, margin: float):
    return np.amax(train_data_set, axis=0) * margin


# def __points(n):
#     # http: // web.archive.org / web / 20120421191837 /
#     # http: // www.cgafaq.info / wiki / Evenly_distributed_points_on_sphere
#
#     dlong = np.pi * (3 - np.sqrt(5))
#     dz = 2.0 / n
#     long = 0
#     z = 1 - dz / 2
#     points = []
#     for k in range(n):
#         r = np.sqrt(1 - z * z)
#         points.append(np.cos(long) * r)
#         points.append(np.sin(long) * r)
#         points.append(z)
#         z = z - dz
#         long = long + dlong
#     return points


def bounding_sphere(n: int, train_data_set: np.array, dim: int, r=1, margin: float=2.0):
    x0 = cartesian(n, r=r, dim=dim)
    # x0 = CMAESAlgorithm.__points(n)
    dividers = np.tile(scale_factor(train_data_set=train_data_set, margin=margin), n)
    x0 = x0 / dividers
    return x0

