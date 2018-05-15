import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functools import reduce

from scripts.benchmarks.data_model import DataModel
from scripts.drawer import draw
from scripts.utils.experimentdatabase import Experiment
from scripts.utils.logger import Logger
from scripts.utils.sampler import bounding_sphere
from scripts.utils.clustering import xmeans_clustering
from sklearn.metrics import confusion_matrix

log = Logger(name='cma-es')

def to_str(w: [list, np.ndarray]):
    return reduce((lambda x, y: str(x) + ' ' + str(y)), w)


class CMAESAlgorithm:

    def __init__(self, w0, n_constraints: int, sigma0: float,
                 scaler: [StandardScaler, None], data_model: DataModel, margin: float,
                 x0: np.ndarray = None, satisfies_constraints: [callable, None] = None, clustering: bool=False, seed: int = 404):
        assert len(w0) == n_constraints

        self.__w0 = w0
        self.__x0 = x0
        self.__train_X = data_model.train_set()
        self.__valid_X = data_model.valid_set()
        self.test_X, self.__test_Y = data_model.test_set()
        self.__dimensions = self.__train_X.shape[1]
        self.__n_constraints = n_constraints
        self.__sigma0 = sigma0
        self.__scaler = scaler
        self.__data_model = data_model
        self.__margin = margin
        self.matches_constraints = satisfies_constraints if satisfies_constraints is not None else self.satisfies_constraints
        self.__clustering = clustering
        self.__results = list()
        self.__seed = seed

        if scaler is not None:
            self.__scaler.fit(self.__train_X)
            self.__train_X = self.__scaler.transform(self.__train_X)
            self.__valid_X = self.__scaler.transform(self.__valid_X)
            self.test_X = self.__scaler.transform(self.test_X)

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
        card_b, tp = self.__train_X.shape[0], self.matches_constraints(self.__train_X, w, w0).sum()
        recall = tp / card_b

        # p
        card_p, p = self.__valid_X.shape[0], self.matches_constraints(self.__valid_X, w, w0).sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y
        f = -f
        log.info("tp: {},\trecall: {},\tp: {},\t pr_y: {},\t\tf: {}".format(tp, p, recall, pr_y, f))
        return f

    def best_results(self):
        final_results = dict()
        y_pred = np.ones(self.__test_Y.shape)
        y_valid_pred = np.ones(self.__test_Y.shape)

        for result in self.__results:
            assert isinstance(result, cma.CMAEvolutionStrategy)
            w = result.best.x
            w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
            w0 = w[-1:]
            w = w[:-1]

            y_pred = y_pred * self.matches_constraints(self.test_X, w, w0)
            y_valid_pred = y_valid_pred * self.matches_constraints(self.__valid_X, w, w0)

        # confusion matrix
        y_true = self.__test_Y
        tn, fp, fn, tp = confusion_matrix(y_true=y_true.astype(int), y_pred=y_pred.astype(int)).ravel()

        # tp
        card_b, tp = self.test_X.shape[0], y_pred.sum()
        recall = tp / card_b

        # p
        card_p, p = self.__valid_X.shape[0], y_valid_pred.sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y
        f = -f

        # final results
        final_results['tn'] = tn
        final_results['tp'] = tp
        final_results['fp'] = fp
        final_results['fn'] = fn
        final_results['f'] = f
        return final_results

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

    def __draw_results(self, w, title=None):
        if self.__valid_X.shape[1] > 4:
            return

        w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        names = ['x_{}'.format(x) for x in np.arange(self.__valid_X.shape[1])]
        data = self.__valid_X if self.__scaler is None else self.__scaler.inverse_transform(self.__valid_X)

        valid = pd.DataFrame(data=data, columns=names)
        valid['valid'] = pd.Series(data=self.matches_constraints(self.__valid_X, w, w0), name='valid')

        train = self.__train_X if self.__scaler is None else self.__scaler.inverse_transform(self.__train_X)
        train = pd.DataFrame(data=train, columns=names)
        train['valid'] = pd.Series(data=self.matches_constraints(self.__train_X, w, w0), name='valid')

        if valid.shape[1] == 3:
            draw.draw2dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        elif valid.shape[1] == 4:
            draw.draw3dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        else:
            pass

    def experiment(self, experiment: Experiment):
        _n = len(self.clusters)

        if self.__clustering:
            for i, cluster in enumerate(self.clusters, start=1):
                log.debug("Started analyzing cluster: {}/{}".format(i, _n))
                self.__train_X = cluster
                cma_es = self.__cma_es()
                self.__results.append(cma_es)
                log.debug("Finished analyzing cluster: {}/{}".format(i, _n))
        else:
            log.debug("Started analyzing train dataset")
            cma_es = self.__cma_es()
            self.__results.append(cma_es)
            log.debug("Finished analyzing train dataset")

        try:
            experiment['seed'] = self.__seed
            experiment['n_constraints'] = self.__n_constraints
            experiment['clusters'] = len(self.clusters)
            experiment['clustering'] = self.__clustering
            experiment['margin'] = self.__margin
            experiment['standardized'] = self.__scaler is not None
            experiment['sigma'] = self.__sigma0
            experiment['name'] = self.__data_model.benchmark_model.name
            experiment['k'] = self.__data_model.benchmark_model.k
            experiment['n'] = self.__dimensions
            final_results = self.best_results()
            experiment['tp'] = int(final_results['tp'])
            experiment['tn'] = int(final_results['tn'])
            experiment['fp'] = int(final_results['fp'])
            experiment['fn'] = int(final_results['fn'])
            experiment['f'] = final_results['f']

            for i, es in enumerate(self.__results):
                es: cma.CMAEvolutionStrategy = es

                W_start = self.split_w(es.x0, split_w=True)
                W = self.split_w(es.best.x, split_w=True)

                cluster = experiment.new_child_data_set('cluster_{}'.format(i))
                cluster['w_start'] = to_str(W_start[0])
                cluster['w0_start'] = to_str(W_start[1])
                cluster['w'] = to_str(W[0])
                cluster['w0'] = to_str(W[1])
                cluster['f'] = es.best.f

        except Exception as e:
            experiment['error'] = e
        finally:
            experiment.save()


    @property
    def sql_params(self):
        return (self.__n_constraints, len(self.clusters), self.__margin, self.__sigma0, self.__data_model.benchmark_model.k,
                 self.__data_model.benchmark_model.i, self.__seed,
                 self.__data_model.benchmark_model.name, self.__clustering, self.__scaler is not None)

# n = 5
# w0 = np.repeat(1, n)
# x0 = None
# scaler = None
# model = DataModel(model=Ball(i=2, B=[1]))
#
# algorithm = CMAESAlgorithm(n_constraints=n, w0=w0, sigma0=1, data_model=model,
#                            scaler=scaler, margin=1, x0=x0, clustering=False,
#                            satisfies_constraints=model.benchmark_model.benchmark_objective_function)
# algorithm.experiment()
#
# args = dict(n_constraints=1, w0=w0, sigma0=1, data_model=model, scaler=scaler, margin=1, x0=x0, clustering=False,
#             satisfies_constraints=None)
