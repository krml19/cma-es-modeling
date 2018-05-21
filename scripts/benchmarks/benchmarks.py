from scripts.benchmarks.cube import Cube
from scripts.benchmarks.ball import Ball
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.simplex import Simplex
from scripts.drawer import draw
from scripts.benchmarks.data_model import DataModel
from scripts.utils.logger import Logger
import pandas as pd
log = Logger(name='benchmarks')


def generate_model(model_type: [Simplex, Cube, Ball], i, B):
    model = model_type(i=i, B=B)
    model.generate_datasets()


for n in range(2, 8):
    for k in range(1, 3):
        for model in [Ball, Simplex, Cube]:
            generate_model(model_type=model, i=n, B=k*[1])



#
# generate_model(model_type=Simplex, i=7, B=[1])
#
# for model in [Simplex, Ball, Cube]:
#     generate_model(model_type=model, i=7, B=2 * [1])
