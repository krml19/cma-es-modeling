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
                 db: str = 'experiments', draw: bool = False):
        data_model = DataModel(name=model_name, k=k, n=n, seed=seed)

        self.__n_constraints = cg.generate(constraints_generator, n)
        self.__w0 = np.repeat(1, self.__n_constraints)
        self.__x0 = x0
        log.debug('Creating train X')
        self.__train_X = data_model.train_set()
        log.debug('Creating valid X')
        self.__valid_X = data_model.valid_set()
        log.debug('Creating test X, Y')
        self.test_X, self.__test_Y = data_model.test_set()
        log.debug('Finished creating datasets')
        self.__dimensions = self.__train_X.shape[1]
        self.__constraints_generator = constraints_generator

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

        if self.__scaler is not None:
            self.__scaler.fit(self.__train_X)
            self.__train_X = self.__scaler.transform(self.__train_X)
            self.__valid_X = self.__scaler.transform(self.__valid_X)
            self.test_X = self.__scaler.transform(self.test_X)

        if clustering_k_min:
            self.clusters = [self.__train_X[x] for x in xmeans_clustering(self.__train_X, kmin=clustering_k_min)]
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
        log.debug(w)
        log.debug(w0)

        # recall
        card_b, tp = self.__train_X.shape[0], self.matches_constraints(self.__train_X, w, w0).sum()
        recall = tp / card_b

        # p
        card_p, p = self.__valid_X.shape[0], self.matches_constraints(self.__valid_X, w, w0).sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y if recall > 0.0 else -1.0 / pr_y
        f = -f
        # log.info("tp: {},\trecall: {},\tp: {},\t pr_y: {},\t\tf: {}".format(tp, recall, p, pr_y, f))
        return f

    def best_results(self):
        final_results = dict()
        y_pred = np.zeros(self.__test_Y.shape)
        y_valid_pred = np.zeros(self.__test_Y.shape)

        for result in self.__results:
            assert isinstance(result, cma.CMAEvolutionStrategy)
            w = result.best.x
            w = np.reshape(w, newshape=(self.__n_constraints, -1)).T
            w0 = w[-1:]
            w = w[:-1]

            y_pred = y_pred + self.matches_constraints(self.test_X, w, w0)
            y_valid_pred = y_valid_pred + self.matches_constraints(self.__valid_X, w, w0)
        y_pred = y_pred > 0
        y_valid_pred = y_valid_pred > 0

        # confusion matrix
        y_true = self.__test_Y
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred.astype(int)).ravel()

        # Based on: https://en.wikipedia.org/wiki/Precision_and_recall
        recall = tp / (tp + fn)

        # p
        card_p = self.__valid_X.shape[0]
        p = y_valid_pred.sum()

        # p_y
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / pr_y
        f = -f

        y_true_sum = y_true.sum()
        ratio = y_pred.sum() / y_true_sum if y_true_sum > 0 else 0
        log.info('Y pred: {}, true: {}, ratio: {}'.format(y_pred.sum(), y_true.sum(), ratio))
        log.info('Final f: {}'.format(f))
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
            x0 = bounding_sphere(n=self.__n_constraints, train_data_set=self.__train_X, dim=self.__dimensions, margin=self.__margin)
        else:
            x0 = self.__x0

        x0 = self.__expand_initial_w(x0=x0)
        f = self.__objective_function(np.array(x0))
        if self.draw:
            self.__draw_results(np.array(x0), title='Initial solution: {}'.format(f))

        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.__sigma0, inopts={'seed': self.__seed, 'maxiter': int(1e4)})

        # FIXME: Add initial solution?
        W = es.ask()
        es.tell(W, [self.__objective_function(w) for w in W])

        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.__objective_function(w) for w in W])
            # es.logger.add()

            # es.disp()  # by default sparse, see option verb_disp

            # log.debug(es.result)

        # log.debug("Best: {}, w: {}".format(es.best.f, es.best.x))
        if self.draw:
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

        train = self.test_X if self.__scaler is None else self.__scaler.inverse_transform(self.test_X)
        train = pd.DataFrame(data=train, columns=names)
        train['valid'] = pd.Series(data=self.matches_constraints(self.test_X, w, w0), name='valid')
        # train['valid'] = self.__test_Y
        if valid.shape[1] == 3:
            draw.draw2dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        elif valid.shape[1] == 4:
            draw.draw3dmodel(df=valid, train=train, constraints=np.split(w, self.__n_constraints, axis=1), title=title, model=self.__data_model.benchmark_model.name)
        else:
            pass

    def experiment(self):
        _n = len(self.clusters)
        start = time.process_time()
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
        self.time_delta = time.process_time() - start



        database = Database(database_filename='{}.sqlite'.format(self.db))
        experiment = database.new_experiment()

        try:
            experiment['benchmark_mode'] = self.benchmark_mode
            experiment['seed'] = self.__seed
            experiment['n_constraints'] = self.__n_constraints
            experiment['constraints_generator'] = self.__constraints_generator
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
            experiment['time'] = self.time_delta
            experiment['test_positives'] = int(self.__test_Y.sum())

            for i, es in enumerate(self.__results):
                es = es

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
            log.info("[Error] Cannot process: {}".format(self.sql_params))
        finally:
            experiment.save()
            log.info("Finished: {} in: {}".format(self.sql_params, str(self.time_delta)))

    @property
    def sql_params(self):
        return (self.__n_constraints, self.__margin, self.__sigma0, self.__data_model.benchmark_model.k,
                 self.__data_model.benchmark_model.i, self.__seed,
                 self.__data_model.benchmark_model.name, self.__clustering, self.__scaler)


# n = 3
# seed = 4
# algorithm = CMAESAlgorithm(constraints_generator=cg.f_2pn.__name__, sigma0=0.1, k=1,
#                            scaler=None, margin=1.1, clustering_k_min=0, model_name='simplex', n=n, seed=seed, draw=True)
# algorithm.experiment()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        argv1 = sys.argv[1].split(';')
        args = dict(e.split(":") for e in argv1)
        args['k'] = int(args['k'])
        args['n'] = int(args['n'])
        args['sigma0'] = float(args['sigma0'])
        args['scaler'] = args['scaler'] == True
        args['margin'] = float(args['margin'])
        args['clustering_k_min'] = int(args['clustering_k_min'])
        args['seed'] = int(args['seed'])
        args['benchmark_mode'] = args['benchmark_mode'] == True

        algorithm = CMAESAlgorithm(**args)
        algorithm.experiment()
