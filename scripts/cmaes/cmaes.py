import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scripts.utils.sampler import bounding_sphere
from scripts.benchmarks.cube import Cube
from scripts.benchmarks.simplex import Simplex

from scripts.drawer import draw
from scripts.utils.logger import Logger

log = Logger(name='cma-es')


class CMAESAlgorithm:

    def __init__(self, train_X: np.ndarray, valid_X: np.ndarray, w0, n_constraints: int, sigma0: float,
                 scaler: [StandardScaler, None], model_type: str, seed: int=77665, margin: float = 2.0, x0: np.ndarray = None):
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
        if scaler is not None:
            self.scaler.fit(train_X)
            self.train_X = self.scaler.transform(train_X)
            self.valid_X = self.scaler.transform(valid_X)

    def objective_function(self, w):
        w = np.reshape(w, newshape=(self.n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        def _objective_function_helper(X):
            x = np.matmul(X, w)
            x = x <= np.sign(w0)
            _len = x.shape[0]
            _sum = x.prod(axis=1).sum()
            return _len, _sum

        # recall
        card_b, tp = _objective_function_helper(self.train_X)
        recall = tp / card_b

        # p
        card_p, p = _objective_function_helper(self.valid_X)

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

        def __get_valid_points(X: np.ndarray):
            x = np.matmul(X, w)
            x = x <= np.sign(w0)
            x = x.prod(axis=1)
            return pd.Series(data=x, name='valid')

        names = ['x_{}'.format(x) for x in np.arange(self.valid_X.shape[1])]
        data = self.valid_X if self.scaler is None else self.scaler.inverse_transform(self.valid_X)

        valid = pd.DataFrame(data=data, columns=names)
        valid['valid'] = __get_valid_points(self.valid_X)
        train = self.train_X if self.scaler is None else self.scaler.inverse_transform(self.train_X)
        train = pd.DataFrame(data=train, columns=names)
        train['valid'] = __get_valid_points(self.train_X)

        if valid.shape[1] == 3:
            draw.draw2dmodel(df=valid, train=train, constraints=np.split(w, self.n_constraints, axis=1), title=title, model=self.model_type)
        elif valid.shape[1] == 4:
            draw.draw3dmodel(df=valid, train=train, constraints=np.split(w, self.n_constraints, axis=1), title=title, model=self.model_type)
        else:
            pass


def data_sets(model: str):
    # load train data
    _train_X = pd.read_csv('data/train_1e5/{}.csv'.format(model), nrows=int(1e5))
    _train_X = _train_X.values

    # load valid data
    _valid_X = pd.read_csv('data/validation/{}.csv'.format(model), nrows=int(1e5))
    _valid_X = _valid_X.values

    return _train_X, _valid_X


# load data
model_type = 'simplex3_0'
train_X, valid_X = data_sets(model_type)

# # run algorithm
model = Simplex(i=3)
x0: np.ndarray = model.optimal_bounding_sphere()
w0: np.ndarray = model.optimal_w0()
n = model.optimal_n_constraints()
scaler = None

# n = 4
# w0 = np.repeat(1, 4)
# x0 = None
# scaler = StandardScaler()

algorithm = CMAESAlgorithm(train_X=train_X, valid_X=valid_X, n_constraints=n, w0=w0, sigma0=1, model_type=model_type,
                           scaler=scaler, margin=1, x0=x0)
algorithm.cma_es()
