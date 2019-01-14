import subprocess
import psutil
import time
import os
# import db
import sqlite3


class ProcessPool:
    def __init__(self, max_processes=psutil.cpu_count(logical=False)):
        self.queue = []
        self.max_processes = max_processes

    def execute(self, cmd, arguments, log_filename):
        self.wait_until_less_than(self.max_processes)
        log = open(log_filename + ".log", "w")
        self.queue.append(dict(process=subprocess.Popen([cmd, arguments], stdout=log, stderr=log), log=log))

    def wait_until_less_than(self, x):
        while True:
            for p in self.queue:
                p["process"].poll()
                if p["process"].returncode is not None:
                    p["log"].close()
                    self.queue.remove(p)

            if len(self.queue) < x:
                break

            time.sleep(0.1)

    def close(self):
        self.wait_until_less_than(1)


class SlurmPool:
    def __init__(self):
        try:
            os.makedirs("./scripts")
        except:
            None
        self.run = open("./run.sh", "w")
        self.run.write("#!/bin/bash\n")
        self.run.write("export LD_LIBRARY_PATH=./gurobi702/linux64/lib/\n")

    def execute(self, cmd, arguments, script_filename):
        sbatch = open("./scripts/%s.sh" % script_filename, "w")
        sbatch.write("#!/bin/bash\n")
        sbatch.write("#SBATCH -p lab-ci,lab-43,lab-44\n")
        sbatch.write("#SBATCH -x lab-al-9\n")
        sbatch.write("#SBATCH -c 1 --mem=1475\n")
        sbatch.write("#SBATCH -t 22:00:00\n")
        sbatch.write("#SBATCH -Q\n")
        # sbatch.write("#SBATCH --nice\n")
        sbatch.write("export GRB_LICENSE_FILE=~/gurobi-$(hostname).lic\n")
        sbatch.write("date\n")
        sbatch.write("hostname\n")
        sbatch.write("echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH\n")
        sbatch.write("echo GRB_LICENSE_FILE=$GRB_LICENSE_FILE\n")
        sbatch.write("echo %s %s\n" % (cmd, arguments))
        sbatch.write("srun %s %s && srun rm \"./scripts/%s.sh\"\n" % (cmd, arguments, script_filename))
        sbatch.close()

        self.run.write("sbatch \"./scripts/%s.sh\"\n" % script_filename)

    def close(self):
        self.run.close()

        ps = subprocess.Popen(["/bin/sh", "./run.sh"])
        ps.wait()


class CompleteDetector:
    def __init__(self, filename="./statistics.sqlite"):
        # self.db = db.prepare_connection(filename)
        # self.no_db = False

        # cursor = self.db.cursor()
        # try:
        #     cursor.execute("CREATE TEMP VIEW IF NOT EXISTS experimentStatistics AS "
        #                    "SELECT *, "
        #                    "(in_problem || benchmarkVariableCount) AS problem, "
        #                    "CASE WHEN in_quadratic IS NULL THEN '' WHEN in_quadratic=1 THEN 'quadratic' END AS instruction "
        #                    "FROM experiments ")
        #
        #     cursor.execute("CREATE INDEX IF NOT EXISTS experimentsIn_nameIn_seed ON experiments(in_name, in_seed)")

        # except sqlite3.OperationalError as e:
        #     self.no_db = True
        #     print(e)
        pass

    def is_completed(self, params):
        # if self.no_db:
        #     return False
        # cursor = self.db.cursor()
        # cursor.execute("SELECT COUNT(*) FROM experimentStatistics "
        #                "WHERE in_name=:method "
        #                "AND ((problem=:benchmark COLLATE NOCASE AND in_samples=:sample) OR in_problem=:problem COLLATE NOCASE) "
        #                "AND instruction=:instruction "
        #                "AND in_seed=:seed", params)
        # data = cursor.fetchall()
        # # print(data[0][0])
        # return data[0][0] > 0
        return False


def main():
    # seeds = range(0, 16)
    seeds = range(0, 30)
    benchmarks = []
    benchmarks += ["Ball3", "Ball4", "Ball5", "Ball6", "Ball7"]
    benchmarks += ["Simplex3", "Simplex4", "Simplex5", "Simplex6", "Simplex7"]
    benchmarks += ["Cube3", "Cube4", "Cube5", "Cube6", "Cube7"]
    problems = []
    # problems += ["Rice", "Rice-nonoise", "Rice-nonoise2"]
    samples = []
    # samples += [10, 20, 30, 40, 50]
    samples += [100, 200, 300, 400]
    # samples += [100, 200, 300]
    # samples += [1000]
    # samples = [300]
    # instructions = ["", "quadratic"]
    instructions = ["quadratic"]
    methods = {
        # "400x50x3": "PopulationSize=400 MaxGenerations=50 LambdaMuRatio=3",
        # "200x100x3": "PopulationSize=200 MaxGenerations=100 LambdaMuRatio=3",
        # "200x50x6": "PopulationSize=200 MaxGenerations=50 LambdaMuRatio=6",
        # "100x100x6": "PopulationSize=100 MaxGenerations=100 LambdaMuRatio=6",

        # "400x50x3S": "PopulationSize=400 MaxGenerations=50 LambdaMuRatio=3 StandardizeData=True",
        # "200x100x3S": "PopulationSize=200 MaxGenerations=100 LambdaMuRatio=3 StandardizeData=True",
        # "200x50x6S": "PopulationSize=200 MaxGenerations=50 LambdaMuRatio=6 StandardizeData=True",
        # "100x100x6S": "PopulationSize=100 MaxGenerations=100 LambdaMuRatio=6 StandardizeData=True",

        # "400x50x3R": "CM=0.5 HR=0.5 PopulationSize=400 MaxGenerations=50 LambdaMuRatio=3",
        # "200x100x3R": "CM=0.5 HR=0.5 PopulationSize=200 MaxGenerations=100 LambdaMuRatio=3",
        # "200x50x6R": "CM=0.5 HR=0.5 PopulationSize=200 MaxGenerations=50 LambdaMuRatio=6",
        # "100x100x6R": "CM=0.5 HR=0.5 PopulationSize=100 MaxGenerations=100 LambdaMuRatio=6",

        "400x50x3RS": "CM=0.5 HR=0.5 PopulationSize=400 MaxGenerations=50 LambdaMuRatio=3 StandardizeData=True",
        # "200x100x3RS": "CM=0.5 HR=0.5 PopulationSize=200 MaxGenerations=100 LambdaMuRatio=3 StandardizeData=True",
        # "200x50x6RS": "CM=0.5 HR=0.5 PopulationSize=200 MaxGenerations=50 LambdaMuRatio=6 StandardizeData=True",
        # "100x100x6RS": "CM=0.5 HR=0.5 PopulationSize=100 MaxGenerations=100 LambdaMuRatio=6 StandardizeData=True",
    }

    #pool = ProcessPool()
    pool = SlurmPool()
    detector = CompleteDetector()

    total_runs = 0
    executed_runs = 0

    for seed in seeds:
        for instruction in instructions:
            for problem in [dict(type="benchmark", name=b, sample=s) for b in benchmarks for s in samples] + [dict(type="problem", name=p) for p in problems]:
                for (method, methodParams) in methods.items():

                    total_runs += 1

                    params = dict(
                        seed=seed,
                        sample=problem["sample"] if "sample" in problem else -1,
                        method=method,
                        method_params=methodParams,
                        instruction=instruction)

                    if problem["type"] == "benchmark":
                        params["benchmark"] = problem["name"]
                        params["problem"] = None
                    else:
                        params["benchmark"] = None
                        params["problem"] = problem["name"]

                    if detector.is_completed(params):
                        continue

                    executed_runs += 1

                    params2 = params.copy()
                    params2["instruction"] = instruction + "=True" if instruction != "" else ""
                    params2["problem"] = "benchmark=Modeling.Common.Benchmarks." + problem["name"] if problem["type"] == "benchmark" else "problem=" + problem["name"] + ".csv"
                    params2["sample"] = "feasibleSamples=" + str(problem["sample"]) if "sample" in problem else ""

                    cmd = "Release/Modeling.MP.exe name=%(method)s %(method_params)s seed=%(seed)d %(problem)s %(sample)s linear=True %(instruction)s synthesizer=Modeling.MP.ES.ESOneClassSynthesizer RemoveRedundantConstraints=False output=statistics.sqlite" % params2
                    log = "%(benchmark)s %(sample)d %(method)s %(instruction)s seed%(seed)d" % params
                    pool.execute("mono", cmd, log)

    print("Executing runs: %d Completed runs: %d Total runs: %d" % (executed_runs, total_runs - executed_runs, total_runs))
    pool.close()

if __name__ == '__main__':
    main()
