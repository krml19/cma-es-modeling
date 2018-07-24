from cube import Cube
from ball import Ball
from simplex import Simplex
from logger import Logger

log = Logger(name='benchmarks')


def generate_model(model_type: [Simplex, Cube, Ball], i, B):
    model = model_type(i=i, B=B)
    model.generate_datasets()


for n in range(2, 8):
    for k in range(1, 3):
        for model in [Ball, Simplex, Cube]:
            generate_model(model_type=model, i=n, B=k*[1])
