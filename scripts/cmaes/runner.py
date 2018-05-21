from scripts.cmaes.cmaes import CMAESAlgorithm
from sklearn.preprocessing import StandardScaler
import numpy as np
from multiprocessing import Pool
from scripts.utils.experimentdatabase import Database
from scripts.utils.logger import Logger

log = Logger(name='runner')


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

    @property
    def sql(self):
        return "select count(*) from experiments where n_constraints=? and margin=? and sigma=? and k=? and n=? and seed=? and name=? and clustering=? and standardized=?"

    def check_if_table_is_empty(self, database: Database):
        check_if_table_exists_query = "SELECT count(*) FROM experiments"
        check_if_table_exists = len(database.engine.execute(check_if_table_exists_query).fetchone()) > 1
        return check_if_table_exists

    def filter_algorithms(self, experiments: list, database: Database) -> list:
        if not self.check_if_table_is_empty(database=database):
            return experiments

        seq_of_params = [self.convert_to_sql_params(experiment) for experiment in experiments]
        db_experiments = [database.engine.execute(self.sql, params).fetchone()[0] for params in seq_of_params]
        filtered = list()

        for exists, algorithm in zip(db_experiments, experiments):
            if exists == 0:
                filtered.append(algorithm)
            else:
                log.info("Experiment already exists.")
        return filtered

    def convert_to_sql_params(self, algorithm_params: dict):
        return (algorithm_params['n_constraints'],
                algorithm_params['margin'],
                algorithm_params['sigma0'],
                algorithm_params['k'],
                algorithm_params['n'],
                algorithm_params['seed'],
                algorithm_params['model_name'],
                algorithm_params['clustering_k_min'],
                algorithm_params['scaler'] is not None)

    def data_source(self, constraints_generator: callable = f_2n, sigma0: float = 1,
                    margin: float = 1.1, scaler: [StandardScaler, None] = None,
                    clustering_k_min: int = 0, benchmark_mode: bool = False, db: str = 'experiments', experiment_n: int = 1):

        experiments = []

        # FIXME: Changes ranges
        for k in range(1, 3):
            for n in range(2, 4):
                for model in ['cube']:
                # for model in ['ball', 'simplex', 'cube']:
                    for seed in range(1, 3):
                        inopts = dict()
                        inopts['n_constraints'] = constraints_generator(n)
                        inopts['w0'] = np.repeat(1, constraints_generator(n))
                        inopts['sigma0'] = sigma0
                        inopts['k'] = k
                        inopts['n'] = n
                        inopts['scaler'] = scaler
                        inopts['margin'] = margin
                        inopts['clustering_k_min'] = clustering_k_min
                        inopts['seed'] = seed
                        inopts['model_name'] = model
                        inopts['benchmark_mode'] = benchmark_mode
                        inopts['db'] = db
                        inopts['experiment_n'] = experiment_n
                        experiments.append(inopts)
        return experiments

    def experiments_1(self) -> list:
        return [self.data_source(scaler=scaler) for scaler in [None, StandardScaler()]]

    def experiments_1_train(self) -> list:
        return [self.data_source(scaler=scaler, db='train') for scaler in [None, StandardScaler()]]

    def experiments_2(self) -> list:
        return [self.data_source(constraints_generator=constraints_generator) for
                       constraints_generator in [f_2n, f_2n2, f_n3, f_2pn]]

    def experiments_3(self) -> list:
        return [self.data_source(clustering_k_min=kmin) for kmin in [0, 1, 2]]

    def experiments_4(self) -> list:
        return [self.data_source(sigma0=sigma) for sigma in [0.5, 1, 1.5]]

    def experiments_5(self) -> list:
        return [self.data_source(margin=margin) for margin in [0.9, 1, 1.1]]

    def benchmarks(self) -> list:
        return [self.data_source(benchmark_mode=True, db='benchmarks')]

    def run_instance(self, inopts: dict):
        algorithm = CMAESAlgorithm(**inopts)
        algorithm.experiment()

    def run(self, experiments: list):

        experiments = flat(experiments)
        db = experiments[0]['db'] + '.sqlite'
        database = Database(database_filename=db)
        experiments = self.filter_algorithms(experiments, database=database)

        pool = Pool(processes=4)  # start 4 worker processes
        pool.map(self.run_instance, experiments)


runner = AlgorithmRunner()
experiments = runner.experiments_1()
runner.run(experiments)
