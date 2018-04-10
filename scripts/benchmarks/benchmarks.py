from scripts.benchmarks.cube import Cube
from scripts.benchmarks.ball import Ball
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.simplex import Simplex
from scripts.drawer import draw


def generate_model(model_type: BenchmarkModel, i=3, d=2.7, rows=5000):
    model = model_type(i=i, d=d, rows=rows)
    df = model.generate_df()
    draw.draw2d(df=df, selected=[1, 0])
    model.save(df)


def generate_validation_dataset(model_type: BenchmarkModel, i=3, d=2.7, rows=int(1e5)):
    model = model_type(i=i, d=d, rows=rows)
    model.generate_validation_dataset()


# generate_model(model_type=Cube)
# generate_model(model_type=Ball)
# generate_model(model_type=Simplex)

generate_validation_dataset(model_type=Ball, i=2)