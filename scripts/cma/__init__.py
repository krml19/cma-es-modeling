import cma

cma.CMAOptions('tol')  # display 'tolerance' termination options
cma.CMAOptions('verb') # display verbosity options
res = cma.fmin(cma.ff.tablet, 15 * [1], 1)
es = cma.CMAEvolutionStrategy(15 * [1], 1).optimize(cma.ff.tablet)
help(es.result)
res[0], es.result[0]  # best evaluated solution
res[5], es.result[5]  # mean solution, presumably better with noise