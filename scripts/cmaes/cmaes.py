import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functools import reduce

from scripts.benchmarks.ball import Ball
from scripts.drawer import draw
from scripts.utils.experimentdatabase import Database, DatabaseException
from scripts.utils.logger import Logger
from scripts.utils.sampler import bounding_sphere
from scripts.utils.clustering import xmeans_clustering

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


def to_str(w: [list, np.ndarray]):
    return reduce((lambda x, y: str(x) + ' ' + str(y)), w)


class CMAESAlgorithm:

    def __init__(self, train_X: np.ndarray, valid_X: np.ndarray, w0, n_constraints: int, sigma0: float,
                 scaler: [StandardScaler, None], model_type: str, seed: int=77665, margin: float = 2.0,
                 x0: np.ndarray = None, objective_func: callable=satisfies_constraints, clustering: bool=False):
        assert train_X.shape[1] == valid_X.shape[1]
        assert len(w0) == n_constraints

        self.__w0 = w0
        self.__x0 = x0
        self.__train_X = train_X
        self.__valid_X = valid_X
        self.__dimensions = train_X.shape[1]
        self.__n_constraints = n_constraints
        self.__sigma0 = sigma0
        self.__scaler = scaler
        self.__seed = seed
        self.__model_type = model_type
        self.__margin = margin
        self.__objective_func = objective_func

        # FIXME: Check if validation set should be scaled only once
        if scaler is not None:
            self.__scaler.fit(train_X)
            self.__train_X = self.__scaler.transform(train_X)
            self.__valid_X = self.__scaler.transform(valid_X)

        if clustering:
            self.clusters = [self.__train_X[x] for x in xmeans_clustering(self.__train_X)]
        else:
            self.clusters = [self.__train_X]

    def __objective_function(self, w):
        w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        # recall
        card_b, tp = self.__train_X.shape[0], self.__objective_func(self.__train_X, w, w0).sum()
        recall = tp / card_b

        # p
        card_p, p = self.__valid_X.shape[0], self.__objective_func(self.__valid_X, w, w0).sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y
        f = -f
        log.info("tp: {},\trecall: {},\tp: {},\t pr_y: {},\t\tf: {}".format(tp, p, recall, pr_y, f))
        return f

    def __confusion_matrix(self, w):
        w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        # recall
        card_b = self.__train_X.shape[0]
        tp = self.__objective_func(self.__train_X, w, w0).sum()
        tn = card_b - tp
        recall = tp / card_b

        # p
        card_p, p = self.__valid_X.shape[0], self.__objective_func(self.__valid_X, w, w0).sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y
        f = -f
        log.info("tp: {},\trecall: {},\tp: {},\t pr_y: {},\t\tf: {}".format(tp, p, recall, pr_y, f))
        return tp

    def __expand_initial_w(self, x0: np.ndarray):
        x0 = np.split(x0, self.__n_constraints)
        w0 = self.__w0
        _x0 = list()
        for xi0, wi0 in zip(x0, w0):
            _x0.append(np.append(xi0, wi0))
        return np.concatenate(_x0)

    def __cma_es(self):
        if self.__x0 is None:
            x0: np.ndarray = bounding_sphere(n=self.__n_constraints, train_data_set=self.__train_X, dim=self.__dimensions, margin=self.__margin)
        else:
            x0 = self.__x0

        x0 = self.__expand_initial_w(x0=x0)
        f = self.__objective_function(np.array(x0))
        self.__draw_results(np.array(x0), title='Initial solution: {}'.format(f))

        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.__sigma0, inopts={'seed': self.__seed, 'maxiter': int(1e4)})

        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.__objective_function(w) for w in W])
            es.logger.add()

            es.disp()  # by default sparse, see option verb_disp

            log.debug(es.result)

        log.info("Best: {}, w: {}".format(es.best.f, es.best.x))
        self.__draw_results(es.best.x, title='Best solution: {}'.format(es.best.f))
        # es.plot()
        return es

    def split_w(self, w, split_w=False):
        w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]
        if split_w:
            w = np.split(w, self.__n_constraints, axis=1)
        return np.concatenate(w).flatten(), np.concatenate(w0)

    def cma_es(self):
        _n = len(self.clusters)
        results = list()

        for i, cluster in enumerate(self.clusters, start=1):
            log.debug("Started analyzing cluster: {}/{}".format(i, _n))
            self.__train_X = cluster
            cma_es = self.__cma_es()
            results.append(cma_es)
            log.debug("Finished analyzing cluster: {}/{}".format(i, _n))

        database = Database(database_filename='experiments.sqlite')
        experiment = database.new_experiment()

        try:
            experiment['seed'] = self.__seed
            experiment['n_constraints'] = self.__n_constraints
            experiment['clusters'] = len(self.clusters)
            experiment['dimensions'] = self.__dimensions
            experiment['margin'] = self.__margin
            experiment['standardized'] = self.__scaler is not None

            f = 0
            for i, es in enumerate(results):
                es: cma.CMAEvolutionStrategy = es

                W_start = self.split_w(es.x0, split_w=True)
                W = self.split_w(es.best.x, split_w=True)

                cluster = experiment.new_child_data_set('cluster_{}'.format(i))
                cluster['w_start'] = to_str(W_start[0])
                cluster['w0_start'] = to_str(W_start[1])
                cluster['w'] = to_str(W[0])
                cluster['w0'] = to_str(W[1])
                cluster['f'] = es.best.f
                f += es.best.f

            experiment['f'] = f
        except Exception as e:
            experiment['error'] = e
        finally:
            experiment.save()

    def __draw_results(self, w, title=None):
        w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        names = ['x_{}'.format(x) for x in np.arange(self.__valid_X.shape[1])]
        data = self.__valid_X if self.__scaler is None else self.__scaler.inverse_transform(self.__valid_X)

        valid = pd.DataFrame(data=data, columns=names)
        valid['valid'] = pd.Series(data=self.__objective_func(self.__valid_X, w, w0), name='valid')

        train = self.__train_X if self.__scaler is None else self.__scaler.inverse_transform(self.__train_X)
        train = pd.DataFrame(data=train, columns=names)
        train['valid'] = pd.Series(data=self.__objective_func(self.__train_X, w, w0), name='valid')

        if valid.shape[1] == 3:
            draw.draw2dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__model_type)
        elif valid.shape[1] == 4:
            draw.draw3dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__model_type)
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
                           scaler=scaler, margin=1, x0=x0, objective_func=benchmark_ball_objective_function, clustering=False)
algorithm.cma_es()
