import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from functools import reduce

from data_model import DataModel
import draw
from experimentdatabase import Database
from logger import Logger
from sampler import bounding_sphere
from clustering import xmeans_clustering
from sklearn.metrics import confusion_matrix
import constraints_generator as cg
import sys
import time

log = Logger(name='cma-es')


def to_str(w: [list, np.ndarray]):
    return reduce((lambda x, y: str(x) + ' ' + str(y)), w)


class CMAESAlgorithm:

    def __init__(self, constraints_generator: str, sigma0: float,
                 scaler: bool, model_name: str, k: int, n: int, margin: float,
                 x0: np.ndarray = None, benchmark_mode: bool = False, clustering_k_min: int=0, seed: int = 404,
                 db: str = 'experiments', draw: bool = False, max_iter: int = int(5e2)):
        data_model = DataModel(name=model_name, k=k, n=n, seed=seed)

        self.__n_constraints = cg.generate(constraints_generator, n)
        self.__w0 = np.repeat(1, self.__n_constraints)
        self.__x0 = x0
        log.debug('Creating train X')
        self.__train_X = data_model.train_set()
        log.debug('Creating valid X')
        self.__valid_X = data_model.valid_set()
        log.debug('Finished creating datasets')
        self.__dimensions = self.__train_X.shape[1]
        self.__constraints_generator = constraints_generator
        self.__test_X, self.__test_Y = None, None
        self.__sigma0 = sigma0
        self.__scaler = StandardScaler() if scaler else None
        self.__data_model = data_model
        self.__margin = margin
        self.matches_constraints = data_model.benchmark_model.benchmark_objective_function if benchmark_mode else self.satisfies_constraints
        self.__clustering = clustering_k_min
        self.__results = list()
        self.__seed = seed
        self.db = db
        self.benchmark_mode = benchmark_mode
        self.draw = draw
        self.time_delta = None
        self.current_cluster = None
        self.max_iter = max_iter

        if self.__scaler is not None:
            self.__scaler.fit(self.__train_X)
            self.__train_X = self.__scaler.transform(self.__train_X)
            self.__valid_X = self.__scaler.transform(self.__valid_X)

        if self.__clustering:
            self.clusters = xmeans_clustering(self.__train_X, kmin=clustering_k_min, visualize=False)

    def satisfies_constraints(self, X: np.ndarray, w: np.ndarray, w0: np.ndarray) -> np.ndarray:
        x = np.matmul(X, w)
        x = x <= np.sign(w0)
        return x.prod(axis=1)

    def __objective_function(self, w):
        w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
        w0 = w[-1:]
        w = w[:-1]

        # recall
        card_b, tp = self.current_cluster.shape[0], self.matches_constraints(self.current_cluster, w, w0).sum()
        recall = tp / card_b

        # p
        card_p, p = self.__valid_X.shape[0], self.matches_constraints(self.__valid_X, w, w0).sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y if recall > 0.0 else -1.0 / pr_y
        f = -f
        log.debug("tp: {},\trecall: {},\tp: {},\t pr_y: {},\t\tf: {}".format(tp, recall, p, pr_y, f))
        return f

    def best(self, X: np.ndarray, Y: np.ndarray, V: np.ndarray):
        final_results = dict()
        y_pred = np.zeros(X.shape[0])
        y_valid_pred = np.zeros(V.shape[0])

        for result in self.__results:
            w = result[0]
            w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
            w0 = w[-1:]
            w = w[:-1]

            y_pred = y_pred + self.matches_constraints(X, w, w0)
            y_valid_pred = y_valid_pred + self.matches_constraints(V, w, w0)
        y_pred = y_pred > 0
        y_valid_pred = y_valid_pred > 0

        # confusion matrix
        res = confusion_matrix(y_true=Y, y_pred=y_pred.astype(int)).ravel()
        if len(res) == 4:
            tn, fp, fn, tp = res
        else:
            tn, fp, fn, tp = 0, 0, 0, res[0]


        # Based on: https://en.wikipedia.org/wiki/Precision_and_recall
        recall = tp / (tp + fn)

        # p
        card_p = V.shape[0]
        p = y_valid_pred.sum()

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
            log.debug("Bounding sphere")
            x0 = bounding_sphere(n=self.__n_constraints, train_data_set=self.current_cluster, dim=self.__dimensions, margin=self.__margin)
        else:
            x0 = self.__x0
        log.debug("Expanding")
        x0 = self.__expand_initial_w(x0=x0)
        f = self.__objective_function(np.array(x0))
        if self.draw:
            self.__draw_results(x0)
        res = cma.fmin(self.__objective_function, x0=x0, sigma0=self.__sigma0,
                       options={'seed': self.__seed, 'maxiter': self.max_iter, 'tolfun': 1e-1, 'timeout': 60 * 60}, restart_from_best=True, eval_initial_x=True)
        if self.draw:
            self.__draw_results(res[0])

        return res

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

        train = self.current_cluster if self.__scaler is None else self.__scaler.inverse_transform(self.current_cluster)
        train = pd.DataFrame(data=train, columns=names)
        train['valid'] = pd.Series(data=self.matches_constraints(self.current_cluster, w, w0), name='valid')
        # train['valid'] = self.__test_Y
        if valid.shape[1] == 3:
            draw.draw2dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        elif valid.shape[1] == 4:
            draw.draw3dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        else:
            pass

    def experiment(self):

        start = time.process_time()
        if self.__clustering:
            _n = len(self.clusters)
            for i, cluster in enumerate(self.clusters, start=1):
                log.debug("Started analyzing cluster: {}/{}".format(i, _n))
                self.current_cluster = self.__train_X[cluster]
                # if self.__scaler:
                #     self.current_cluster = self.__scaler.transform(self.current_cluster)
                cma_es = self.__cma_es()
                self.__results.append(cma_es)
                log.debug("Finished analyzing cluster: {}/{}".format(i, _n))
        else:
            log.debug("Started analyzing train dataset")
            self.current_cluster = self.__train_X
            cma_es = self.__cma_es()
            self.__results.append(cma_es)
            log.debug("Finished analyzing train dataset")
        self.time_delta = time.process_time() - start


        log.debug('Creating test X, Y')
        self.__test_X, self.__test_Y = self.__data_model.test_set()
        if self.__scaler is not None:
            self.__test_X = self.__scaler.transform(self.__test_X)

        best_train = self.best(X=self.__train_X, V=self.__valid_X, Y=np.ones(self.__train_X.shape[0]))
        best_test = self.best(X=self.__test_X, V=self.__valid_X, Y=self.__test_Y)

        database = Database(database_filename='{}.sqlite'.format(self.db))
        experiment = database.new_experiment()

        try:
            experiment['benchmark_mode'] = self.benchmark_mode
            experiment['seed'] = self.__seed
            experiment['n_constraints'] = self.__n_constraints
            experiment['constraints_generator'] = self.__constraints_generator
            experiment['clusters'] = len(self.clusters) if self.__clustering else 0
            experiment['clustering'] = self.__clustering
            experiment['margin'] = self.__margin
            experiment['standardized'] = self.__scaler is not None
            experiment['sigma'] = self.__sigma0
            experiment['name'] = self.__data_model.benchmark_model.name
            experiment['k'] = self.__data_model.benchmark_model.k
            experiment['n'] = self.__dimensions
            experiment['max_iter'] = self.max_iter

            experiment['tp'] = int(best_test['tp'])
            experiment['tn'] = int(best_test['tn'])
            experiment['fp'] = int(best_test['fp'])
            experiment['fn'] = int(best_test['fn'])
            experiment['f'] = best_test['f']

            experiment['train_tp'] = int(best_train['tp'])
            experiment['train_tn'] = int(best_train['tn'])
            experiment['train_fp'] = int(best_train['fp'])
            experiment['train_fn'] = int(best_train['fn'])
            experiment['train_f'] = best_train['f']

            experiment['time'] = self.time_delta

            for i, es in enumerate(self.__results):
                es = es

                W_start = self.split_w(es[8].x0, split_w=True)
                W = self.split_w(es[0], split_w=True)

                cluster = experiment.new_child_data_set('cluster_{}'.format(i))
                cluster['w_start'] = to_str(W_start[0])
                cluster['w0_start'] = to_str(W_start[1])
                cluster['w'] = to_str(W[0])
                cluster['w0'] = to_str(W[1])
                cluster['f'] = es[1]

        except Exception as e:
            experiment['error'] = e
            log.error("Cannot process: {}".format(self.sql_params))
            print(e)
        finally:
            experiment.save()
            log.info("Finished: {} in: {}".format(self.sql_params, str(self.time_delta)))
            log.info("Train: {}".format(best_train))
            log.info("Test: {}".format(best_test))

    @property
    def sql_params(self):
        return (self.__constraints_generator, self.__n_constraints, self.__margin, self.__sigma0, self.__data_model.benchmark_model.k,
                 self.__data_model.benchmark_model.i, self.__seed,
                 self.__data_model.benchmark_model.name, self.__clustering, self.__scaler is not None)


# algorithm = CMAESAlgorithm(constraints_generator=cg.f_2n.__name__, sigma0=2, k=2,
#                            scaler=True, margin=1.1, clustering_k_min=2, model_name='cube', n=2, seed=4, draw=False)
# algorithm.experiment()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        argv1 = sys.argv[1].split(';')
        args = dict(e.split(":") for e in argv1)
        args['k'] = int(args['k'])
        args['n'] = int(args['n'])
        args['sigma0'] = float(args['sigma0'])
        args['scaler'] = args['scaler'] == 'True'
        args['margin'] = float(args['margin'])
        args['clustering_k_min'] = int(args['clustering_k_min'])
        args['seed'] = int(args['seed'])
        args['benchmark_mode'] = args['benchmark_mode'] == 'True'

        algorithm = CMAESAlgorithm(**args)
        algorithm.experiment()
