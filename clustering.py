from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.utils import timedcall

import numpy as np
from logger import Logger
from sklearn.cluster import KMeans

# Based on: https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/examples/xmeans_examples.py
def xmeans_clustering(data: np.ndarray, kmin: [int, None] = 1, kmax: [int, None] = 20, tolerance: float = 0.025,
                      criterion: enumerate = splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore: bool = True,
                      logger=Logger(name='clustering'),
                      visualize: bool = False) -> np.ndarray:

    # Initial centers - KMeans algorithm
    kmeans = KMeans(n_clusters=kmin)
    kmeans.fit(data)
    initial_centers = kmeans.cluster_centers_

    # X-Means algorithm
    xmeans_instance = xmeans(data=data, initial_centers=initial_centers, kmax=kmax, tolerance=tolerance,
                             criterion=criterion, ccore=ccore)
    (ticks, _) = timedcall(xmeans_instance.process)

    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()

    criterion_string = "UNKNOWN"
    if criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION:
        criterion_string = "BAYESIAN INFORMATION CRITERION"
    elif criterion == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH:
        criterion_string = "MINIMUM NOISELESS DESCRIPTION_LENGTH"

    if logger is not None:
        logger.debug("Initial centers: {},\n Execution time: {},\n Number of clusters: {},\n criterion: {}".format(
            initial_centers is not None, ticks, len(clusters), criterion_string))

    if visualize:
        visualizer = cluster_visualizer()
        visualizer.set_canvas_title(criterion_string)
        visualizer.append_clusters(clusters, data)
        visualizer.append_cluster(centers, None, marker='*')
        visualizer.show()

    return clusters
