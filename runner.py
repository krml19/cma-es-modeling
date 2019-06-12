from cmaes import CMAESAlgorithm
from multiprocessing import Pool
from experimentdatabase import Database
from logger import Logger
import constraints_generator as cg
import os
import subprocess
from functools import reduce
from frozendict import frozendict
import psutil

log = Logger(name='runner')


def flat(l):
    return [item for sublist in l for item in sublist]


class SlurmPool:
    def __init__(self):
        try:
            os.makedirs("./scripts")
        except:
            None
        self.run = open("./run.sh", "w")
        self.run.write("#!/bin/bash\n")
        self.run.write("export PATH=\"/home/inf116360/anaconda3/bin:$PATH\"\n")

    def execute(self, cmd, arguments, script_filename):
        sbatch = open("./scripts/%s.sh" % script_filename, "w")
        sbatch.write("#!/bin/bash\n")
        sbatch.write("#SBATCH -p idss-student")
        sbatch.write("#SBATCH -c 1 --mem=1475\n")
        sbatch.write("#SBATCH -t 24:00:00\n")
        sbatch.write("#SBATCH -Q\n")
        sbatch.write("date\n")
        sbatch.write("hostname\n")
        sbatch.write("echo %s %s\n" % (cmd, arguments))
        sbatch.write("srun %s %s && srun rm \"./scripts/%s.sh\"\n" % (cmd, arguments, script_filename))
        sbatch.close()

        self.run.write("sbatch \"./scripts/%s.sh\"\n" % script_filename)

    def close(self):
        self.run.close()

        ps = subprocess.Popen(["/bin/sh", "./run.sh"])
        ps.wait()


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
        log.info("Number of existing experiments: {}/{}".format(existing, len(experiments)))
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

    def data_source(self, scaler: bool = True,
                    constraints_generator: callable = cg.f_n3,
                    clustering_k_min: int = 2,
                    sigma0: float = 0.125,
                    margin: float = 1.0,
                    benchmark_mode: bool = False,
                    seeds: range = range(0, 30), K: range = range(1, 3), N: range = range(2, 8), train_sample: int = 500, models=['ball', 'simplex', 'cube']):

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

    def experiments_1(self, seeds: range = range(0, 30), K: range = range(1, 3), N: range = range(2, 8), models=['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(seeds=seeds, K=K, N=N,
                                 scaler=scaler,
                                 constraints_generator=cg.f_2np2,
                                 clustering_k_min=1,
                                 sigma0=0.5,
                                 margin=1.0,
                                 train_sample=300,
                                 models=models) for scaler in [True, False]]

    def experiments_2(self, seeds: range = range(0, 30), K: range = range(1, 3), N: range = range(2, 8), models=['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(seeds=seeds, K=K, N=N,
                                 scaler=True,
                                 constraints_generator=constraints_generator,
                                 clustering_k_min=1,
                                 sigma0=0.5,
                                 margin=1.0,
                                 train_sample=300,
                                 models=models) for constraints_generator in [cg.f_2n, cg.f_2np2, cg.f_n3, cg.f_2pn]]  # cg.f_n1,

    def experiments_3(self, seeds: range = range(0, 30), K: range = range(1, 3), N: range = range(2, 8), models=['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(seeds=seeds, K=K, N=N,
                                 scaler=True,
                                 constraints_generator=cg.f_2pn,
                                 clustering_k_min=kmin,
                                 sigma0=0.5,
                                 margin=1.0,
                                 train_sample=300,
                                 models=models) for kmin in [0, 1, 2]]

    def experiments_4(self, seeds: range = range(0, 30), K: range = range(1, 3), N: range = range(2, 8), models=['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(seeds=seeds, K=K, N=N,
                                 scaler=True,
                                 constraints_generator=cg.f_2pn,
                                 clustering_k_min=2,
                                 sigma0=sigma,
                                 margin=1.0,
                                 train_sample=300,
                                 models=models) for sigma in [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]]

    def experiments_5(self, seeds: range = range(0, 30), K: range = range(1, 3), N: range = range(2, 8), models=['ball', 'simplex', 'cube']) -> list:
        return [self.data_source(seeds=seeds, K=K, N=N,
                                 scaler=True,
                                 constraints_generator=cg.f_2pn,
                                 clustering_k_min=2,
                                 sigma0=0.03125,
                                 margin=margin,
                                 train_sample=300,
                                 models=models) for margin in [0.9, 1, 1.1]]

    def experiments_6(self, seeds: range = range(0, 30), N: range = range(2, 8), K=range(1, 3), models=['ball', 'simplex', 'cube']) -> list:
        return [
            self.data_source(seeds=seeds, K=K, N=N,
                             scaler=True,
                             constraints_generator=cg.f_2pn,
                             clustering_k_min=2,
                             sigma0=0.03125,
                             margin=1.0,
                             train_sample=ts,
                             models=models) for ts in [500, 400, 100, 200, 300]]

    def experiments_case_study(self, seeds: range = range(0, 30), N=[8], K=[1], models=['case_study']) -> list:
        return [
            self.data_source(seeds=seeds, K=K, N=N,
                             scaler=True,
                             constraints_generator=cg.f_2pn,
                             clustering_k_min=2,
                             sigma0=0.03125,
                             margin=1.0,
                             train_sample=1022,
                             models=models)]

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
            'case_study': self.experiments_case_study(seeds=seeds),
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
            log.info("Number of existing experiments in experiment {}: {}/{}".format(i, len(filtered), len(experiments)))

    def run_instance(self, inopts: dict):
        algorithm = CMAESAlgorithm(**inopts)
        algorithm.experiment()

    def run(self, experiments: list):
        experiments = flat(experiments)
        db = 'experiments.sqlite'
        database = Database(database_filename=db)
        experiments = self.filter_algorithms(experiments, database=database)

        cpus = 3  # psutil.cpu_count(logical=False)
        pool = Pool(processes=cpus)  # start worker processes
        pool.map(self.run_instance, experiments, 1)

    def run_slurm(self, experiments: list):
        experiments = set(flat(experiments))
        db = 'experiments.sqlite'
        database = Database(database_filename=db)
        experiments = self.filter_algorithms(experiments, database=database)

        pool = SlurmPool()
        for experiment in experiments:
            try:
                mapped = list(map(lambda item: item[0] + ':' + str(item[1]), experiment.items()))
                arguments = "\"" + reduce(lambda key, value: key + ';' + value, mapped) + "\""
                pool.execute(cmd='python', arguments='cmaes.py {}'.format(arguments),
                             script_filename=str(self.convert_to_sql_params(experiment)))
            except:
                print(experiment)

        pool.close()


if __name__ == '__main__':
    runner = AlgorithmRunner()
    seeds = range(0, 15)
    models = ['ball', 'cube', 'simplex']
    experiments = flat([
        runner.experiments_1(seeds=seeds, N=range(3, 8), K=range(1, 3), models=models),
        runner.experiments_2(seeds=seeds, N=range(3, 8), K=range(1, 3), models=models),
        runner.experiments_3(seeds=seeds, N=range(3, 8), K=range(1, 3), models=models),
        runner.experiments_4(seeds=seeds, N=range(3, 8), K=range(1, 3), models=models),
        runner.experiments_5(seeds=seeds, N=range(3, 8), K=range(1, 3), models=models),
        runner.experiments_6(seeds=seeds, N=range(3, 8), K=range(1, 3), models=models),
        # runner.experiments_case_study(seeds=seeds)
    ])
    # experiments = runner.experiments_1(seeds=seeds)
    runner.run(experiments)
    # runner.run_slurm(experiments)
