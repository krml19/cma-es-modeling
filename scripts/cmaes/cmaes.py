import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scripts.utils.sampler import bounding_sphere
from scripts.benchmarks.cube import Cube
from scripts.benchmarks.simplex import Simplex
from scripts.benchmarks.ball import Ball

from scripts.drawer import draw
from scripts.utils.logger import Logger

log = Logger(name='cma-es')


def benchmark_ball_objective_function(X: np.ndarray, w: np.ndarray, w0: np.ndarray) -> np.ndarray:
    dim = X.shape[1]
    d: float = 2.7
    ii = np.arange(1, dim + 1)
    X = ((X - ii) ** 2).sum(axis=1)
    X: np.ndarray = X <= np.repeat(d, dim).prod()
    return X


def satisfies_constraints(X: np.ndarray, w: np.ndarray, w0: np.ndarray) -> np.ndarray:
    x = np.matmul(X, w)
    x = x <= np.sign(w0)
    return x.prod(axis=1)


class CMAESAlgorithm:

    def __init__(self, train_X: np.ndarray, valid_X: np.ndarray, w0, n_constraints: int, sigma0: float,
                 scaler: [StandardScaler, None], model_type: str, seed: int=77665, margin: float = 2.0,
                 x0: np.ndarray = None, objective_func: callable=satisfies_constraints):
        assert train_X.shape[1] == valid_X.shape[1]
        assert len(w0) == n_constraints

        self.w0 = w0
        self.x0 = x0
        self.train_X = train_X
        self.valid_X = valid_X
        self.dimensions = train_X.shape[1]
        self.n_constraints = n_constraints
        self.sigma0 = sigma0
        self.es = None
        self.scaler = scaler
        self.seed = seed
        self.model_type = model_type
        self.margin = margin
        self.objective_func = objective_func

        if scaler is not None:
            self.scaler.fit(train_X)
            self.train_X = self.scaler.transform(train_X)
            self.valid_X = self.scaler.transform(valid_X)

    def objective_function(self, w):
        w = np.reshape(w, newshape=(self.n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        # recall
        card_b, tp = self.train_X.shape[0], self.objective_func(self.train_X, w, w0).sum()
        recall = tp / card_b

        # p
        card_p, p = self.valid_X.shape[0], self.objective_func(self.valid_X, w, w0).sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y
        f = -f
        log.info("tp: {},\trecall: {},\tp: {},\t pr_y: {},\t\tf: {}".format(tp, p, recall, pr_y, f))
        return f

    def __expand_initial_w(self, x0: np.ndarray):
        x0 = np.split(x0, self.n_constraints)
        w0 = self.w0
        _x0 = list()
        for xi0, wi0 in zip(x0, w0):
            _x0.append(np.append(xi0, wi0))
        return np.concatenate(_x0)

    def cma_es(self):
        if self.x0 is None:
            x0: np.ndarray = bounding_sphere(n=self.n_constraints, train_data_set=self.train_X, dim=self.dimensions, margin=self.margin)
        else:
            x0 = self.x0

        x0 = self.__expand_initial_w(x0=x0)
        f = self.objective_function(np.array(x0))
        self.draw_results(np.array(x0), title='Initial solution: {}'.format(f))

        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.sigma0, inopts={'seed': self.seed, 'maxiter': int(1e4)})

        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.objective_function(w) for w in W])
            es.logger.add()

            es.disp()  # by default sparse, see option verb_disp

            log.debug(es.result)

        log.info("Best: {}, w: {}".format(es.best.f, es.best.x))
        self.draw_results(es.best.x, title='Best solution: {}'.format(es.best.f))
        # es.plot()
        self.es = es

    def draw_results(self, w, title=None):
        w = np.reshape(w, newshape=(self.n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        names = ['x_{}'.format(x) for x in np.arange(self.valid_X.shape[1])]
        data = self.valid_X if self.scaler is None else self.scaler.inverse_transform(self.valid_X)

        valid = pd.DataFrame(data=data, columns=names)
        valid['valid'] = pd.Series(data=self.objective_func(self.valid_X, w, w0), name='valid')

        train = self.train_X if self.scaler is None else self.scaler.inverse_transform(self.train_X)
        train = pd.DataFrame(data=train, columns=names)
        train['valid'] = pd.Series(data=self.objective_func(self.train_X, w, w0), name='valid')

        if valid.shape[1] == 3:
            draw.draw2dmodel(df=valid, train=train, constraints=np.split(w, self.n_constraints, axis=1), title=title, model=self.model_type)
        elif valid.shape[1] == 4:
            draw.draw3dmodel(df=valid, train=train, constraints=np.split(w, self.n_constraints, axis=1), title=title, model=self.model_type)
        else:
            pass


def data_sets(model: str):
    # load train data
    _train_X = pd.read_csv('data/train/{}.csv'.format(model), nrows=int(1e5))
    _train_X = _train_X.values

    # load valid data
    _valid_X = pd.read_csv('data/validation/{}.csv'.format(model), nrows=int(1e5))
    _valid_X = _valid_X.values

    return _train_X, _valid_X


# load data
model_type = 'ball2_0'
train_X, valid_X = data_sets(model_type)

# # run algorithm
model = Ball(i=2)
x0: np.ndarray = model.optimal_bounding_sphere()
w0: np.ndarray = model.optimal_w0()
n = model.optimal_n_constraints()
scaler = None

# n = 4
# w0 = np.repeat(1, 4)
# x0 = None
# scaler = StandardScaler()

algorithm = CMAESAlgorithm(train_X=train_X, valid_X=valid_X, n_constraints=n, w0=w0, sigma0=1, model_type=model_type,
                           scaler=scaler, margin=1, x0=x0, objective_func=benchmark_ball_objective_function)
algorithm.cma_es()
