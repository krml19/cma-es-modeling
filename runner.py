from cmaes import CMAESAlgorithm
from multiprocessing import Pool
from experimentdatabase import Database
from logger import Logger
import constraints_generator as cg
from frozendict import frozendict
import psutil

log = Logger(name='runner')


def flat(l):
    return [item for sublist in l for item in sublist]


class AlgorithmRunner:

    @property
    def sql(self):
        return "SELECT count(*) FROM experiments WHERE constraints_generator=? AND margin=? AND sigma=? AND k=? AND n=? AND seed=? AND name=? AND clustering=? AND standardized=? AND (train_tp + train_fn)=?"

    def check_if_table_is_empty(self, database: Database):
        check_if_table_exists = database.engine.execute("SELECT count(*) FROM main.experiments").fetchone()[0]
        return check_if_table_exists > 0

    def filter_algorithms(self, experiments: iter, database: Database) -> list:
        if not self.check_if_table_is_empty(database=database):
            return experiments

        seq_of_params = [self.convert_to_sql_params(experiment) for experiment in experiments]
        db_experiments = [database.engine.execute(self.sql, params).fetchone()[0] for params in seq_of_params]
        filtered = list()

        existing = 0
        for exists, algorithm in zip(db_experiments, experiments):
            if exists == 0:
                filtered.append(algorithm)
            else:
                existing = existing + 1
        log.info("Number of exisiting experiments: {}/{}".format(existing, len(experiments)))
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
                algorithm_params['scaler'],
                algorithm_params['train_sample'])

    def data_source(self, constraints_generator: callable = cg.f_n3, sigma0: float = 0.5,
                    margin: float = 1.0, scaler: bool = True, clustering_k_min: int = 2, benchmark_mode: bool = False,
                    seeds: range = range(0, 30), K: range=range(1,3), N: range = range(2, 8), train_sample: int = 500, models: list = ['ball', 'simplex', 'cube']):

        experiments = []
        for seed in seeds:
            for k in K:
                for n in N:
                    for model in models:
                        inopts = frozendict({
                            'constraints_generator': constraints_generator.__name__,
                            'sigma0': sigma0,
                            'k': k,
                            'n': n,
                            'scaler': scaler,
                            'margin': margin,
                            'clustering_k_min': clustering_k_min,
                            'seed': seed,
                            'model_name': model,
                            'benchmark_mode': benchmark_mode,
                            'train_sample': train_sample
                        })

                        experiments.append(inopts)
        return experiments

    def experiments_1(self, seeds: range = range(0, 30), K: range=range(1,3), N: range = range(2, 8), models: list = ['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(scaler=scaler, seeds=seeds, K=K, N=N, constraints_generator=cg.f_2np2,
                                 clustering_k_min=0, sigma0=1.0, margin=1.0, models=models) for scaler in [True, False]]

    def experiments_2(self, seeds: range = range(0, 30), K: range=range(1,3), N: range = range(2, 8), models: list = ['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(constraints_generator=constraints_generator, seeds=seeds, K=K, N=N, clustering_k_min=0,
                                 sigma0=1.0, margin=1.0, models=models) for constraints_generator in [cg.f_2n, cg.f_2np2, cg.f_n3, cg.f_2pn]]

    def experiments_3(self, seeds: range = range(0, 30), K: range=range(1,3), N: range = range(2, 8), models: list = ['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(clustering_k_min=kmin, seeds=seeds, K=K, N=N, sigma0=1.0, margin=1.0, models=models) for kmin in [0, 1, 2]]

    def experiments_4(self, seeds: range = range(0, 30), K: range=range(1,3), N: range = range(2, 8), models: list = ['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(sigma0=sigma, seeds=seeds, K=K, N=N, margin=1.0, models=models) for sigma in [0.125, 0.25, 0.5, 1, 2]]

    def experiments_5(self, seeds: range = range(0, 30), K: range=range(1,3), N: range = range(2, 8), models: list = ['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(margin=margin, seeds=seeds, K=K, N=N, models=models) for margin in [0.9, 1, 1.1]]

    def experiments_6(self, seeds: range = range(0, 30), N: range = range(2, 8), K: range=range(1, 3), models: list = ['ball', 'simplex', 'cube']) -> list:
        return [
            self.data_source(seeds=seeds, N=N, train_sample=ts, models=models, K=K) for ts in [100, 200, 300, 400, 500]]

    def benchmarks(self) -> list:
        return [self.data_source(benchmark_mode=True)]

    def experiment(self, key, seeds: range = range(0, 30)):
        return {
            1: self.experiments_1(seeds=seeds),
            2: self.experiments_2(seeds=seeds),
            3: self.experiments_3(seeds=seeds),
            4: self.experiments_4(seeds=seeds),
            5: self.experiments_5(seeds=seeds),
            6: self.experiments_6(seeds=seeds),
            'best': [self.data_source(seeds=seeds)],
            'benchmarks': self.benchmarks()
        }[key]

    def check_existing_experiments(self, seeds: range = range(0, 30)):
        db = 'experiments.sqlite'
        database = Database(database_filename=db)

        for i in range(1, 6):
            experiments = set(flat(self.experiment(i, seeds=seeds)))
            seq_of_params = [self.convert_to_sql_params(experiment) for experiment in experiments]
            db_experiments = [database.engine.execute(self.sql, params).fetchone()[0] for params in seq_of_params]
            filtered = list(filter(lambda t: t[0] > 0, zip(db_experiments, experiments)))
            log.info("Number of exisiting experiments in experiment {}: {}/{}".format(i, len(filtered), len(experiments)))

    def run_instance(self, inopts: dict):
        algorithm = CMAESAlgorithm(**inopts)
        algorithm.experiment()

    def run(self, experiments: list):
        experiments = flat(experiments)
        db = 'experiments.sqlite'
        database = Database(database_filename=db)
        experiments = self.filter_algorithms(experiments, database=database)

        cpus = psutil.cpu_count(logical=False)
        pool = Pool(processes=cpus)  # start worker processes
        pool.map(self.run_instance, experiments)


if __name__ == '__main__':
    runner = AlgorithmRunner()
    seeds = range(0, 30)
    experiments = flat([
        runner.experiments_1(seeds=seeds),
        runner.experiments_2(seeds=seeds),
        runner.experiments_3(seeds=seeds),
        runner.experiments_4(seeds=seeds),
        runner.experiments_5(seeds=seeds),
        runner.experiments_6(seeds=seeds)
        ])
    runner.run(experiments)
