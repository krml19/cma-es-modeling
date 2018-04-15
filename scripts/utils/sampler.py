import numpy as np


def samples(bounds, rows, cols=1, seed=404):
    np.random.seed(seed)
    return np.matrix(
        [np.random.uniform(low=low, high=high, size=(rows, cols)).flatten() for low, high in bounds])


def uniform_samples(low, high, size, seed=404):
    np.random.seed(seed)
    return np.random.uniform(low=low, high=high, size=size)

