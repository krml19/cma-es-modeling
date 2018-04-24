from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans, splitting_type

from pyclustering.utils import timedcall
import pandas as pd
import numpy as np
from scripts.utils.logger import Logger


def xmeans_clustering(data: np.ndarray, initial_centers: np.ndarray = None, kmax: int = 20, tolerance: float = 0.025,
                      criterion: enumerate = splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore: bool = True,
                      logger=Logger(name='clustering'),
                      visualize: bool = True) -> np.ndarray:
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


data: np.ndarray = pd.read_csv('data/train/simplex3_0.csv', nrows=int(1e3)).values
clusters = xmeans_clustering(data=data)
