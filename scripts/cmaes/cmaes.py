import cma


# help(cma)  # "this" help message, use cma? in ipython
# help(cma.fmin)
# help(cma.CMAEvolutionStrategy)
# help(cma.CMAOptions)
# cma.CMAOptions('tol')  # display 'tolerance' termination options
# cma.CMAOptions('verb') # display verbosity options
# res = cma.fmin(cma.ff.tablet, 15 * [1], 1)
# es = cma.CMAEvolutionStrategy(15 * [1], 1).optimize(cma.ff.tablet)
# help(es.result)
# res[0], es.result[0]  # best evaluated solution
# res[5], es.result[5]  # mean solution, presumably better with noise
# cma.plot()

import cma
from cma.fitness_functions import FitnessFunctions
from numpy import array, dot, isscalar, sum
import numpy as np


class Fitness(FitnessFunctions):
    def rosen2(self, x, alpha=1e2):
        """Rosenbrock test objective function"""
        x = [x] if isscalar(x[0]) else x  # scalar into list
        x = np.asarray(x)
        f = [sum(alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in x]
        result = f if len(f) > 1 else f[0]  # 1-element-list into scalar
        return result

# construct an object instance in 4-D, sigma0=1
es = cma.CMAEvolutionStrategy(4 * [1], 1, {'seed':234})
fff = Fitness()
# iterate until termination
while not es.stop():
    X = es.ask()
    es.tell(X, [fff.rosen2(x) for x in X])
    es.disp()  # by default sparse, see option verb_disp

print(es.result)

help(es.result)
