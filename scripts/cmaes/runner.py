from scripts.benchmarks.ball import Ball
from scripts.benchmarks.cube import Cube
from scripts.benchmarks.simplex import Simplex
from scripts.cmaes.cmaes import CMAESAlgorithm
from scripts.benchmarks.data_model import DataModel
from sklearn.preprocessing import StandardScaler
import numpy as np
from multiprocessing import Pool
from scripts.utils.experimentdatabase import Database, Experiment


def flat(l):
    return [item for sublist in l for item in sublist]


def f_2n(n):
    return 2 * n


def f_2n2(n):
    return 2 * n ** n


def f_n3(n):
    return np.power(n, 3)


def f_2pn(n):
    return np.power(2, n)

def pr(x):
    print(x)


class AlgorithmRunner:

    __database = Database(database_filename='experiments.sqlite')

    def sql(self):
        return "select * from experiments where n_constraints=? and clusters=? and margin=? and sigma=? and k=? and n=? and seed=? and name=? and clustering=? and standardized=?"

    def check_if_table_is_empty(self):
        check_if_table_exists_query = "SELECT count(*) FROM experiments"
        check_if_table_exists = len(self.__database.engine.execute(check_if_table_exists_query).fetchone()) > 1
        return check_if_table_exists

    def filter_algorithms(self, algorithms: list) -> list:
        if not self.check_if_table_is_empty():
            return algorithms

        seq_of_params = [algorithm.sql_params for algorithm in algorithms]
        cursor = self.__database.engine.executemany(sql=self.sql(), seq_of_parameters=seq_of_params)
        filtered = list()

        for exists, algorithm in zip(cursor.fetchall(), algorithms):
            if exists == 0:
                filtered.append(algorithm)
        return filtered

    def data_source(self, constraints_generator: callable = f_2n, w0: np.ndarray = np.repeat(1, 4), sigma0: float = 1,
                    margin: float = 1, scaler: [StandardScaler, None] = None, satisfies_constraints: [callable, None] = None,
                    clustering: bool = False):

        experiments = []
        # FIXME: Changes ranges
        for k in range(1, 2):
            for n in range(2, 3):
                for model in [Ball, Simplex, Cube]:
                    for seed in range(1, 31):
                        data_model = DataModel(model=model(B=k * [1], i=n))
                        algorithm = CMAESAlgorithm(n_constraints=constraints_generator(n), w0=w0, sigma0=sigma0,
                                                   data_model=data_model, scaler=scaler, margin=margin,
                                                   clustering=clustering, satisfies_constraints=satisfies_constraints,
                                                   seed=seed)
                        experiments.append(algorithm)
        return experiments

    def experiments_1(self) -> list:
        return [self.data_source(scaler=scaler) for scaler in [None, StandardScaler()]]

    def experiments_2(self) -> list:
        return [self.data_source(constraints_generator=constraints_generator) for
                       constraints_generator in [f_2n, f_2n2, f_n3, f_2pn]]

    def run_instance(self, algorithm: CMAESAlgorithm):
        # algorithm.experiment(experiment=self.__database.new_experiment())
        print(algorithm.sql_params)

    def run(self, experiments: list):
        experiments = flat(experiments)
        experiments = self.filter_algorithms(experiments)
        # for experiment in experiments:
        #     experiment.experiment()

        pool = Pool(processes=4)  # start 4 worker processes
        pool.map(self.run_instance, experiments)  # evaluate "f(10)" asynchronously
        # pool.map(pr, [2, 3])


runner = AlgorithmRunner()
experiments = runner.experiments_1()
runner.run(experiments)
