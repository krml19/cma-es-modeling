import cma
from numpy import isscalar, sum
import numpy as np
import time


class CMAESAlgorithm:

    def __init__(self, X:np.ndarray, P_y=0.7, w_0=1):
        self.P_y = P_y
        self.w_0 = w_0
        self.X = X

    def objective_function(self, w):
        w = [w] if isscalar(w[0]) else w  # scalar into list
        w = np.asarray(w)
        return sum((w - 2) ** 2)

    def cma_es(self):
        # construct an object instance in 2-D, sigma0=1
        dimensions = len(self.X)
        es = cma.CMAEvolutionStrategy(dimensions * [1], 1, {'seed': time.time()})
        # iterate until termination
        while not es.stop():
            W = es.ask()
            es.tell(W, [self.objective_function(w) for w in W])
            es.logger.add()
            # es.plot()
            es.disp()  # by default sparse, see option verb_disp

        print(es.result)


X = np.ones((2, 6))

algorithm = CMAESAlgorithm(X)
algorithm.cma_es()
