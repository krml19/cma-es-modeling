from scripts.cmaes.cmaes import CMAESAlgorithm
from sklearn.preprocessing import StandardScaler
import numpy as np
from multiprocessing import Pool
from scripts.utils.experimentdatabase import Database
from scripts.utils.logger import Logger
import scripts.utils.constraints_generator as cg

log = Logger(name='runner')


def flat(l):
    return [item for sublist in l for item in sublist]


class AlgorithmRunner:

    @property
    def sql(self):
        return "select count(*) from experiments where constraints_generator=? and margin=? and sigma=? and k=? and n=? and seed=? and name=? and clustering=? and standardized=? and experiment_n=?"

    def check_if_table_is_empty(self, database: Database):
        check_if_table_exists = database.engine.execute("select count(*) from main.experiments").fetchone()[0]
        return check_if_table_exists > 0

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
                log.info("Experiment already exists: {}".format(algorithm))
        return filtered

    def convert_to_sql_params(self, algorithm_params: dict):
        return (algorithm_params['constraints_generator'],
                algorithm_params['margin'],
                algorithm_params['sigma0'],
                algorithm_params['k'],
                algorithm_params['n'],
                algorithm_params['seed'],
                algorithm_params['model_name'],
                algorithm_params['clustering_k_min'],
                algorithm_params['scaler'] is not None,
                algorithm_params['experiment_n'])

    def data_source(self, constraints_generator: callable = cg.f_2n, sigma0: float = 1,
                    margin: float = 1.1, scaler: [StandardScaler, None] = None,
                    clustering_k_min: int = 0, benchmark_mode: bool = False, db: str = 'experiments', experiment_n: int = 1):

        experiments = []
        np.std()
        # FIXME: Changes ranges
        for k in range(1, 3):
            for n in range(2, 5):
                for model in ['ball', 'simplex', 'cube']:
                    for seed in range(1, 5):
                        inopts = dict()
                        inopts['constraints_generator'] = constraints_generator.__name__
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
        return [self.data_source(constraints_generator=constraints_generator, experiment_n=2) for
                       constraints_generator in [cg.f_2n, cg.f_2n2, cg.f_n3, cg.f_2pn]]

    def experiments_3(self) -> list:
        return [self.data_source(clustering_k_min=kmin, experiment_n=3) for kmin in [0, 1, 2]]

    def experiments_4(self) -> list:
        return [self.data_source(sigma0=sigma, experiment_n=4) for sigma in [0.125, 0.25, 0.5, 1]]

    def experiments_5(self) -> list:
        return [self.data_source(margin=margin, experiment_n=5) for margin in [0.9, 1, 1.1]]

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
experiments = flat([runner.experiments_1(), runner.experiments_2(), runner.experiments_3(), runner.experiments_4(), runner.experiments_5()])
runner.run(experiments)
