import cma
import numpy as np
import pandas as pd


class CMAESAlgorithm:

    def __init__(self, train_X: np.matrix, valid_X: np.matrix, w_0=1):
        self.w_0 = w_0
        self.train_X = train_X
        self.valid_X = valid_X

    def objective_function(self, w):
        # recall
        b = self.train_X.dot(w)
        tp = (b <= self.w_0).sum()
        card_b = b.shape[1]
        recall = tp / card_b

        # pr_y
        p = self.valid_X.dot(w)
        card_p = p.shape[1]
        p = (p <= self.w_0).sum()
        pr_y = p / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        f = recall ** 2 / pr_y

        return f

    def cma_es(self):
        # construct an object instance in 2-D, sigma0=1
        dimensions = self.train_X.shape[1]
        es = cma.CMAEvolutionStrategy(dimensions * [1], 1, {'seed': 234})

        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.objective_function(w) for w in W])
            es.logger.add()
            # es.plot()
            es.disp()  # by default sparse, see option verb_disp

        print(es.result)


X = np.ones((2, 6))

train_X = pd.read_csv('data/train/cube2_0.csv')
train_X = np.matrix(train_X)

valid_X = pd.read_csv('data/validation/cube2_0.csv')
valid_X = np.matrix(valid_X)


algorithm = CMAESAlgorithm(train_X=train_X, valid_X=valid_X)
algorithm.cma_es()
