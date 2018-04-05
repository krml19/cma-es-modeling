import cma
import numpy as np
import pandas as pd
from scripts.drawer import draw
from scripts.utils import sampler
from scripts.utils.logger import Logger
from sklearn.preprocessing import StandardScaler
import time


log = Logger()


class CMAESAlgorithm:

    def __init__(self, train_X: np.matrix, valid_X: np.matrix, w_0=1, n_constraints=2, sigma0=2, scaler=StandardScaler()):
        self.w_0 = w_0
        self.train_X = train_X
        self.valid_X = valid_X
        self.dimensions = self.train_X.shape[1]
        self.number_of_constraints = n_constraints
        self.sigma0 = sigma0
        self.es = None
        self.scaler = scaler
        if scaler is not None:
            self.scaler.fit(train_X)
            self.train_X = self.scaler.transform(train_X)
            self.valid_X = self.scaler.transform(valid_X)

    def objective_function(self, w):
        # self.draw_results(w)
        w = np.split(w, self.number_of_constraints)

        b_final = np.ones((self.train_X.shape[0],))
        p_final = np.ones((self.valid_X.shape[0],))

        for wi in w:
            # b
            b = self.train_X.dot(wi)
            bf = b <= self.w_0
            b_final = np.multiply(bf, b_final)
            log.debug("b: {}".format(bf))

            # p
            p = self.valid_X.dot(wi)
            p = p <= self.w_0
            p_final = np.multiply(p, p_final)
            log.debug("p: {}".format(p))

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
        log.info("tp: {},\tp: {},\trecall: {},\t pr_y: {},\t\tf: {}".format(tp, p_final, recall, pr_y, f))
        return f

    def mock_cma_es(self):
        mock_x0 = [1 / 3, 0, -1 / 3, 0, 0, 1 / 5, 0, -1 / 5]
        self.draw_results(np.array(mock_x0))

    def cma_es(self):
        # x0 = sampler.uniform_samples(-10, 10, size=(self.dimensions * self.number_of_constraints,))
        # x0 = [2] * (self.dimensions * self.number_of_constraints)
        # x0 = [1/3.7, 0, -1, 0, 0, -1/2, 0, 1/7.4]
        # x0 = [1 / 3.7, 0, -1, 0, 0, 1 / 7.4, 0, -1/2]

        # mock_x0 = [1/3, 0, -1/3, 0, 0, 1/5, 0, -1/5]
        # self.draw_results(np.array(mock_x0))
        # seed = 46463
        # x0 = [-1 / 3.7, 0, 1, 0, 0, 1 / 2, 0, -1 / 7.4]
        # seed = 767657

        mins = 1 / self.train_X.min(axis=0)
        maxs = 1 / self.train_X.max(axis=0)
        min0 = mins[0]
        min1 = mins[1]
        max0 = maxs[0]
        max1 = maxs[1]

        x0 = [min0, 0, max0, 0, 0, min1, 0, max1]
        self.draw_results(np.array(x0))

        seed = time.time()
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.sigma0, inopts={'seed': seed, 'maxiter': 1})

        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.objective_function(w) for w in W])
            es.logger.add()
            # es.plot()
            es.disp()  # by default sparse, see option verb_disp

            log.debug(es.result)

        log.info("Best: {}, w: {}".format(es.best.f, es.best.x))
        self.draw_results(es.best.x)
        self.es = es

    def draw_results(self, w):
        w = np.split(w, self.number_of_constraints)

        p_final = np.ones((self.valid_X.shape[0],))
        for wi in w:
            # p
            p = self.valid_X.dot(wi)
            p = p <= self.w_0
            p_final = np.multiply(p, p_final)

        names = ['x_{}'.format(x) for x in np.arange(self.valid_X.shape[1])]
        data = self.valid_X if self.scaler is None else self.scaler.inverse_transform(self.valid_X)

        df = pd.DataFrame(data=data, columns=names)
        df['valid'] = pd.Series(data=p_final, name='valid')
        draw.draw2d(df, constraints=w)
        draw.draw2d(df)

# load train data
train_X = pd.read_csv('data/train/cube2_0.csv', nrows=500)
train_X = train_X.values

# load valid data
valid_X = pd.read_csv('data/validation/cube2_0.csv', nrows=None)
valid_X = valid_X.values


mock_valid_X = pd.read_csv('data/mock/mock_valid.csv')
mock_valid_X = mock_valid_X.values

# run algorithm
algorithm = CMAESAlgorithm(train_X=train_X, valid_X=valid_X, n_constraints=4)
algorithm.cma_es()

scaler = StandardScaler()
scaler.fit(mock_valid_X)
scaled = scaler.transform(mock_valid_X)
df = pd.DataFrame(data=scaled, columns=['x_xx', 'x_yyy'])
# draw.draw2d(df)
print(df)



