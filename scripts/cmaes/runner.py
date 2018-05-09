from scripts.benchmarks.ball import Ball
from scripts.benchmarks.cube import Cube
from scripts.benchmarks.simplex import Simplex
from scripts.cmaes.cmaes import CMAESAlgorithm
from scripts.benchmarks.data_model import DataModel
from sklearn.preprocessing import StandardScaler
import numpy as np


class AlgorithmRunner:

    @staticmethod
    def f_2n(n):
        return 2 * n

    @staticmethod
    def f_2n2(n):
        return 2 * n ** n

    @staticmethod
    def f_n3(n):
        return np.power(n, 3)

    @staticmethod
    def f_2pn(n):
        return np.power(2, n)

    @staticmethod
    def run(constraints_generator: callable = f_2n, w0: np.ndarray = np.repeat(1, 4), sigma0: float = 1,
            margin: float = 1, scaler: [StandardScaler, None] = None, satisfies_constraints: [callable, None] = None,
            clustering: bool = False):

        for k in range(1, 3):
            for n in range(2, 8):
                for model in [Ball, Simplex, Cube]:
                    for seed in range(1, 31):
                        data_model = DataModel(model=model(B=k * [1], i=n))
                        algorithm = CMAESAlgorithm(n_constraints=constraints_generator(n), w0=w0, sigma0=sigma0,
                                                   data_model=data_model, scaler=scaler, margin=margin,
                                                   clustering=clustering, satisfies_constraints=satisfies_constraints,
                                                   seed=seed)
                        algorithm.experiment()

    @staticmethod
    def run_experiment_1():
        for scaler in [None, StandardScaler()]:
            AlgorithmRunner.run(scaler=scaler)

    @staticmethod
    def run_experiment_2():
        for constraints_generator in [AlgorithmRunner.f_2n, AlgorithmRunner.f_2n2, AlgorithmRunner.f_n3, AlgorithmRunner.f_2pn]:
            AlgorithmRunner.run(constraints_generator=constraints_generator)


