import numpy as np


def samples(bounds, rows, cols=1):
    return np.matrix(
        [np.random.uniform(low=low, high=high, size=(rows, cols)).flatten() for low, high in bounds])


def uniform_samples(low, high, size):
    return np.random.uniform(low=low, high=high, size=size)

