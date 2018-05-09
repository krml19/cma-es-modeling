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


for model in [Cube, Ball, Simplex]:
    # FIXME: change range
    for n in range(5, 8):
        for k in range(1, 3):
            generate_model(model_type=model, i=n, B=k*[1])
