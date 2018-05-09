import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functools import reduce

from scripts.benchmarks.data_model import DataModel
from scripts.drawer import draw
from scripts.utils.experimentdatabase import Database
from scripts.utils.logger import Logger
from scripts.utils.sampler import bounding_sphere
from scripts.utils.clustering import xmeans_clustering
from scripts.benchmarks.ball import Ball

log = Logger(name='cma-es')

def to_str(w: [list, np.ndarray]):
    return reduce((lambda x, y: str(x) + ' ' + str(y)), w)


class CMAESAlgorithm:

    def __init__(self, w0, n_constraints: int, sigma0: float,
                 scaler: [StandardScaler, None], data_model: DataModel, margin: float = 2.0,
                 x0: np.ndarray = None, objective_func: [callable, None] = None, clustering: bool=False):
        assert len(w0) == n_constraints

        self.__w0 = w0
        self.__x0 = x0
        self.__train_X = data_model.train_set()
        self.__valid_X = data_model.valid_set()
        self.__test_X = data_model.test_set()
        self.__dimensions = self.__train_X.shape[1]
        self.__n_constraints = n_constraints
        self.__sigma0 = sigma0
        self.__scaler = scaler
        self.__data_model = data_model
        self.__margin = margin
        self.__objective_func = objective_func if objective_func is not None else self.satisfies_constraints
        self.__clustering = clustering

        if scaler is not None:
            self.__scaler.fit(self.__train_X)
            self.__train_X = self.__scaler.transform(self.__train_X)
            self.__valid_X = self.__scaler.transform(self.__valid_X)
            self.__test_X = self.__scaler.transform(self.__test_X)

        if clustering:
            self.clusters = [self.__train_X[x] for x in xmeans_clustering(self.__train_X)]
        else:
            self.clusters = [self.__train_X]

    def satisfies_constraints(self, X: np.ndarray, w: np.ndarray, w0: np.ndarray) -> np.ndarray:
        x = np.matmul(X, w)
        x = x <= np.sign(w0)
        return x.prod(axis=1)

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
        # TODO: tn, fn, fp
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

        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.__sigma0, inopts={'seed': self.__data_model.benchmark_model.seed, 'maxiter': int(1e4)})

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
            experiment['seed'] = self.__data_model.benchmark_model.seed
            experiment['n_constraints'] = self.__n_constraints
            experiment['clusters'] = len(self.clusters)
            experiment['clustering'] = self.__clustering
            experiment['margin'] = self.__margin
            experiment['standardized'] = self.__scaler is not None
            experiment['name'] = self.__data_model.benchmark_model.name
            experiment['k'] = self.__data_model.benchmark_model.k
            experiment['n'] = self.__dimensions

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
        if self.__valid_X.shape[1] > 4:
            return

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
            draw.draw2dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        elif valid.shape[1] == 4:
            draw.draw3dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        else:
            pass

n = 4
w0 = np.repeat(1, 4)
x0 = None
scaler = None
model = DataModel(model=Ball(i=2, B=[1, 1]))

algorithm = CMAESAlgorithm(n_constraints=n, w0=w0, sigma0=1, data_model=model,
                           scaler=scaler, margin=1, x0=x0, clustering=False, objective_func=model.benchmark_model.benchmark_objective_function)
algorithm.cma_es()
