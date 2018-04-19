from scripts.benchmarks.cube import Cube
from scripts.benchmarks.ball import Ball
from scripts.benchmarks.benchmark_model import BenchmarkModel
from scripts.benchmarks.simplex import Simplex
from scripts.drawer import draw
from scripts.utils.logger import Logger
import pandas as pd
log = Logger(name='benchmarks')


def generate_model(model_type: BenchmarkModel, i=3, d=2.7, rows=int(5e4)):
    model = model_type(i=i, d=d, rows=rows)
    df = model.generate_df(take_only_valid_points=True)
    assert isinstance(df, pd.DataFrame)
    # df = df.apply(lambda x: x**2)
    if i == 2:
        draw.draw2d(df=df, title=model.name)
    elif i == 3:
        draw.draw3d(df=df, title=model.name)

    # model.save(df.head(int(1e5)), path='data/train')



# def generate_validation_dataset(model_type: BenchmarkModel, i=3, d=2.7, rows=int(1e5)):
#     model = model_type(i=i, d=d, rows=rows)
#     model.generate_validation_dataset()

# for model in [Cube, Ball, Simplex]:
#     generate_validation_dataset(model_type=model, i=3)


for model in [Ball]:
    generate_model(model_type=model, i=2)