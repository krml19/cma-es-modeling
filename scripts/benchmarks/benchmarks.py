from scripts.benchmarks.cube import Cube
from scripts.benchmarks.ball import Ball
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.simplex import Simplex
from scripts.drawer import draw
from scripts.benchmarks.data_model import DataModel
from scripts.utils.logger import Logger
import pandas as pd
log = Logger(name='benchmarks')


def generate_model(model_type: BenchmarkModel, i=3, d=2.7, B=[1]):
    model = model_type(i=i, d=d, B=B)
    model.generate_datasets()

for model in [Cube]:
    generate_model(model_type=model, i=2, B=[1, 1])