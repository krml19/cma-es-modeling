from scripts.benchmarks.cube import Cube
from scripts.benchmarks.ball import Ball
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.simplex import Simplex


def generate_model(model_type: BenchmarkModel, i=3, d=2.7, rows=5000):
    model = model_type(i=i, d=d, rows=rows)
    df = model.generate_df()
    model.save(df)


generate_model(model_type=Cube)
generate_model(model_type=Ball)
generate_model(model_type=Simplex)

import logging
logging.basicConfig()