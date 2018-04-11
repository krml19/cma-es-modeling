import cma
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools

from scripts.drawer import draw
from scripts.utils.logger import Logger
from scripts.utils import sampler

log = Logger()


class CMAESAlgorithm:

    def __init__(self, train_X: np.matrix, valid_X: np.matrix, w0=list([1, -1]), n_constraints=2, sigma0=5,
                 scaler=StandardScaler(), model_type='cube'):
        self.w0 = w0
        self.train_X = train_X
        self.valid_X = valid_X
        self.n_constraints = n_constraints
        self.sigma0 = sigma0
        self.es = None
        self.scaler = scaler
        self.model_type = model_type
        if scaler is not None:
            self.scaler.fit(train_X)
            self.train_X = self.scaler.transform(train_X)
            self.valid_X = self.scaler.transform(valid_X)

    def objective_function(self, w):
        # self.draw_results(w)
        w = np.split(w, self.n_constraints)

        b_final = np.ones((self.train_X.shape[0],))
        p_final = np.ones((self.valid_X.shape[0],))

        for wi in w:
            # b
            b = self.train_X.dot(wi)
            # p
            p = self.valid_X.dot(wi)
            for wi0 in self.w0:
                # b
                bi = b <= wi0
                b_final = np.multiply(bi, b_final)
                log.debug("b: {}".format(bi))

                # p
                pi = p <= wi0
                p_final = np.multiply(pi, p_final)
                log.debug("p: {}".format(pi))

        # recall
        card_b = len(b_final)
        tp = b_final.sum()
        recall = tp / card_b

        # p
        card_p = len(p_final)
        p_final = p_final.sum()

        # p_y
        pr_y = p_final / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = (recall ** 2) / (pr_y)
        f = -f
        log.info("tp: {},\trecall: {},\tp: {},\t pr_y: {},\t\tf: {}".format(tp, p_final, recall, pr_y, f))
        return f

    @staticmethod
    def ct(arr, r, decimals):
        a = np.concatenate((np.array([2 * np.pi]), arr))
        si = np.sin(a)
        si[0] = 1
        si = np.cumprod(si)
        co = np.cos(a)
        co = np.roll(co, -1)
        return np.round(si * co * r, decimals=decimals)

    @staticmethod
    def cartesian(n=1, decimals=5, r=1):
        phis = np.arange(n)/n*2*np.pi

        coordinates = [CMAESAlgorithm.ct([phi], r=r, decimals=decimals) for phi in phis]
        flatten = np.array(list(itertools.chain.from_iterable(coordinates)))
        return flatten

    @staticmethod
    def scale_factor(train_data_set: np.array):
        return np.abs(train_data_set).max()

    @staticmethod
    def bounding_sphere(n: int, train_data_set: np.array, r=1, decimals=5):
        x0 = CMAESAlgorithm.cartesian(n, decimals=decimals, r=r)
        x0 = x0 / CMAESAlgorithm.scale_factor(train_data_set)
        return x0

    def cma_es(self):
        x0 = CMAESAlgorithm.bounding_sphere(n=self.n_constraints, train_data_set=self.train_X)
        f = self.objective_function(np.array(x0))
        self.draw_results(np.array(x0), title='Initial solution: {}'.format(f))

        seed = 77665
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.sigma0, inopts={'seed': seed, 'maxiter': int(1e7)})

        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.objective_function(w) for w in W])
            es.logger.add()
            # es.plot()
            es.disp()  # by default sparse, see option verb_disp

            log.debug(es.result)

        log.info("Best: {}, w: {}".format(es.best.f, es.best.x))
        self.draw_results(es.best.x, title='Best solution: {}'.format(es.best.f))
        self.es = es

    def draw_results(self, w, title=None):
        w = np.split(w, self.n_constraints)

        def __valid_points(dataset: np.ndarray):
            p_final = np.ones((dataset.shape[0],))
            for wi in w:
                # p
                p = dataset.dot(wi)
                for wi0 in self.w0:
                    pi = p <= wi0
                    p_final = np.multiply(pi, p_final)
            return pd.Series(data=p_final, name='valid')

        names = ['x_{}'.format(x) for x in np.arange(self.valid_X.shape[1])]
        data = self.valid_X if self.scaler is None else self.scaler.inverse_transform(self.valid_X)

        valid = pd.DataFrame(data=data, columns=names)
        valid['valid'] = __valid_points(self.valid_X)
        train = self.train_X if self.scaler is None else self.scaler.inverse_transform(self.train_X)
        train = pd.DataFrame(data=train, columns=names)
        train['valid'] = __valid_points(self.train_X)

        draw.draw2dmodel(df=valid, train=train, constraints=w, title=title, model=self.model_type)


def data_sets(model_type: str):
    # load train data
    train_X = pd.read_csv('data/train/{}2_0.csv'.format(model_type), nrows=500)
    train_X = train_X.values

    # load valid data
    valid_X = pd.read_csv('data/validation/{}2_0.csv'.format(model_type), nrows=None)
    valid_X = valid_X.values

    return train_X, valid_X


# load data
model_type = 'cube'
train_X, valid_X = data_sets(model_type)

# # run algorithm
algorithm = CMAESAlgorithm(train_X=train_X, valid_X=valid_X, n_constraints=6, w0=[1], sigma0=1, model_type=model_type)
algorithm.cma_es()

