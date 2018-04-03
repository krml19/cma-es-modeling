import cma
import numpy as np
import pandas as pd
from scripts.drawer import draw
from scripts.utils import sampler


class CMAESAlgorithm:

    def __init__(self, train_X: np.matrix, valid_X: np.matrix, w_0=1, number_of_constraints=2, sigma0=2):
        self.w_0 = w_0
        self.train_X = train_X
        self.valid_X = valid_X
        self.dimensions = self.train_X.shape[1]
        self.number_of_constraints = number_of_constraints
        self.sigma0 = sigma0
        self.es = None

    def objective_function(self, w):
        w = np.split(w, self.number_of_constraints)

        b_final = np.ones((1, self.train_X.shape[0]))
        p_final = np.ones((1, self.valid_X.shape[0]))
        print("\n\n")
        for wi in w:
            # b
            b = self.train_X.dot(wi)
            print("b: {}".format(b))
            b = b <= self.w_0
            # print(b_final)
            # print(b)
            b_final = np.multiply(b, b_final)
            # print(b_final)
            # print("\n")

            # p
            p = self.valid_X.dot(wi)
            p = p <= self.w_0
            p_final = np.multiply(p, p_final)
            print("p: {}".format(p))

        # recall
        card_b = b_final.shape[1]
        tp = b_final.sum()
        recall = tp / card_b

        # p
        card_p = p_final.shape[1]
        p_final = p_final.sum()

        # p_y
        pr_y = p_final / card_p
        pr_y = max(pr_y, 1e-6)  # avoid division by 0

        # f
        f = recall ** 2 / pr_y
        f = -f
        print("f: {},\ttp: {},\tp: {},\trecall: {},\t pr_y: {}".format(f, tp, p_final, recall, pr_y))
        return f

    def cma_es(self):
        # construct an object instance in 2-D, sigma0=1
        initial_W = sampler.uniform_samples(-10, 10, size=(self.dimensions * self.number_of_constraints,))
        es = cma.CMAEvolutionStrategy(x0=initial_W, sigma0=self.sigma0, inopts={'seed': 234})

        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.objective_function(w) for w in W])
            es.logger.add()
            # es.plot()
            es.disp()  # by default sparse, see option verb_disp

        print(es.result)
        self.draw_results(es.best.x)
        self.es = es

    def draw_results(self, w):
        w = np.split(w, self.number_of_constraints)

        p_final = np.ones((1, self.valid_X.shape[0]))
        print("\n\n")
        for wi in w:

            # p
            p = self.valid_X.dot(wi)
            p = p <= self.w_0
            p_final = np.multiply(p, p_final)

        names = ['x_{}'.format(x) for x in np.arange(self.valid_X.shape[1])]
        df = pd.DataFrame(data=self.valid_X, columns=names)
        df['valid'] = pd.Series(data=np.array(p_final).flatten(), name='valid')
        draw.draw2d(df)


train_X = pd.read_csv('data/train/cube2_0.csv', nrows=500)
# draw.draw2d(train_X)
train_X = np.matrix(train_X)

valid_X = pd.read_csv('data/validation/cube2_0.csv', nrows=None)
# draw.draw2d(valid_X)
valid_X = np.matrix(valid_X)

algorithm = CMAESAlgorithm(train_X=train_X, valid_X=valid_X, number_of_constraints=8)
algorithm.cma_es()
