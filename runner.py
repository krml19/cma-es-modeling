from cmaes import CMAESAlgorithm
from multiprocessing import Pool
from experimentdatabase import Database
from logger import Logger
import constraints_generator as cg
import os
import subprocess
from functools import reduce

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
        # sbatch.write("#SBATCH -p lab-ci,lab-43,lab-44\n")
        # sbatch.write("#SBATCH -x lab-al-9\n")
        sbatch.write("#SBATCH -p idss-student")
        sbatch.write("#SBATCH -c 1 --mem=1475\n")
        sbatch.write("#SBATCH -t 0:10:00\n")
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
        return "SELECT count(*) FROM experiments WHERE constraints_generator=? AND margin=? AND sigma=? AND k=? AND n=? AND seed=? AND name=? AND clustering=? AND standardized=?"

    def check_if_table_is_empty(self, database: Database):
        check_if_table_exists = database.engine.execute("SELECT count(*) FROM main.experiments").fetchone()[0]
        return check_if_table_exists > 0

    def filter_algorithms(self, experiments: list, database: Database) -> list:
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
                log.info("Experiment already exists: {}".format(algorithm))
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
                algorithm_params['scaler'])

    def data_source(self, constraints_generator: callable = cg.f_2n, sigma0: float = 0.5,
                    margin: float = 1.1, scaler: bool = False,
                    clustering_k_min: int = 0, benchmark_mode: bool = False):

        experiments = []

        for k in range(1, 3):
            for n in range(2, 8):
                for model in ['ball', 'simplex', 'cube']:
                    for seed in range(0, 4):
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
                        experiments.append(inopts)
        return experiments

    def experiments_1(self) -> list:
        return [self.data_source(scaler=scaler) for scaler in [True, False]]

    def experiments_2(self) -> list:
        return [self.data_source(constraints_generator=constraints_generator) for
                constraints_generator in [cg.f_2n, cg.f_2n2, cg.f_n3, cg.f_2pn]]

    def experiments_3(self) -> list:
        return [self.data_source(clustering_k_min=kmin) for kmin in [0, 1, 2]]

    def experiments_4(self) -> list:
        return [self.data_source(sigma0=sigma) for sigma in [0.125, 0.25, 0.5, 1]]

    def experiments_5(self) -> list:
        return [self.data_source(margin=margin) for margin in [0.9, 1, 1.1]]

    def benchmarks(self) -> list:
        return [self.data_source(benchmark_mode=True)]

    def experiment(self, key):
        return {
            1: self.experiments_1(),
            2: self.experiments_2(),
            3: self.experiments_3(),
            4: self.experiments_4(),
            5: self.experiments_5(),
            'benchmarks': self.benchmarks()
        }[key]

    def run_instance(self, inopts: dict):
        try:
            algorithm = CMAESAlgorithm(**inopts)
            algorithm.experiment()
        except:
            log.info("Error: {}".format(inopts))

    def run(self, experiments: list):
        experiments = flat(experiments)
        db = 'experiments.sqlite'
        database = Database(database_filename=db)
        experiments = self.filter_algorithms(experiments, database=database)

        pool = Pool(processes=4)  # start 4 worker processes
        pool.map(self.run_instance, experiments)

    def run_slurm(self, experiments: list):
        experiments = flat(experiments)
        db = 'experiments.sqlite'
        database = Database(database_filename=db)
        experiments = self.filter_algorithms(experiments, database=database)

        pool = SlurmPool()
        for experiment in experiments:
            try:
                mapped = list(map(lambda item: item[0] + ':' + str(item[1]), experiment.items()))
                arguments = "\"" + reduce(lambda key, value: key + ';' + value, mapped) + "\""
                pool.execute(cmd='python', arguments='cma-es {}'.format(arguments),
                             script_filename=str(self.convert_to_sql_params(experiment)))
            except:
                print(experiment)

        pool.close()


if __name__ == '__main__':
    runner = AlgorithmRunner()
    experiments = flat([runner.experiments_1(), runner.experiments_2(), runner.experiments_3(), runner.experiments_4(), runner.experiments_5()])
    # experiments = runner.experiments_1()
    # runner.run(experiments)
    runner.run_slurm(experiments)
