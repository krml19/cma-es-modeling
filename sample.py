from concurrent.futures import ProcessPoolExecutor

import psutil

from Problem import *
import pandas as pd
import os.path
import numpy as np


def main():
    pool = ProcessPoolExecutor(psutil.cpu_count(logical=False))

    try:
        seeds = range(30)
        problems = [Cube, Simplex, Ball]
        for seed in seeds:
            for k in range(1, 3):
                for n in range(2, 8):
                    for p in problems:
                        train_csv = "data/training_%s_%d_%d_%d.csv.xz" % (p.__name__, k, n, seed)
                        test_csv = "data/test_%s_%d_%d_%d.csv.xz" % (p.__name__, k, n, seed)
                        validation_csv = "data/validation_%s_%d_%d_%d.csv.xz" % (p.__name__, k, n, seed)
                        validation2_csv = "data/validation2_%s_%d_%d_%d.csv.xz" % (p.__name__, k, n, seed)
                        pool.submit(sample, 1000, k, n, p, [1], seed, train_csv)
                        pool.submit(sample, 100000, k, n, p, [0, 1], seed + 100, test_csv)
                        pool.submit(sample, 100000, k, n, p, [0, 1], seed + 1000, validation_csv)
                        pool.submit(sample, 100000, k, n, p, [0, 1], seed + 10000, validation2_csv)

    finally:
        pool.shutdown(True)


def sample(count, k, n, p, classes, seed, csv):
    if os.path.isfile(csv):
        return

    print(csv)
    np.random.seed(seed)
    problem = p(k, n)

    instance = problem.sample(count, classes)
    data = instance.data

    variables = ["x%d[%f|%f]" % (i, domain[0], domain[1]) for i, domain in enumerate(problem.domains)]
    if len(classes) > 1:
        variables.append("y")
        data = np.concatenate((data, instance.target[:, np.newaxis]), axis=1)

    frame = pd.DataFrame(data=data, columns=variables)
    frame.to_csv(csv, index=False, compression="xz")


def dataset(count, k, n, p, classes, seed):
    np.random.seed(seed)
    problem = p(k, n)

    instance = problem.sample(count, classes)
    data = instance.data

    if len(classes) > 1:
        data = np.concatenate((data, instance.target[:, np.newaxis]), axis=1)

    return data


if __name__ == '__main__':
    main()
