from models import Simplex, Cube, Ball
from logger import Logger
import file_helper as fh
log = Logger(name='benchmarks')


def generate_model(model_type: [Simplex, Cube, Ball], n, B, seed, rows=500):
    model = model_type(n=n, B=B)
    dataset = model.generate_train_dataset(rows=rows, seed=seed)
    filename = 'training_%s_%d_%d_%d' % (model.name.title(), model.k, model.n, seed)
    fh.write_data_frame(dataset, filename=filename, path=fh.Paths.train.value)

for seed in range(0, 30):
    for n in range(2, 8):
        for k in range(1, 3):
            for model in [Ball, Simplex, Cube]:
                generate_model(model_type=model, n=n, B=k*[1], seed=seed)
