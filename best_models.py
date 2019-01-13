import sqlite3

import numpy as np
from file_helper import write_file, Paths
from sklearn.preprocessing import StandardScaler
from data_model import DataModel
from logger import Logger

log = Logger(name='best-models')

from constraints_generator import generate as n_constraints
get_params_sql = """
select id, constraints_generator, seed, standardized from experiments where name = ? and n=3 and k=? order by f limit 1
"""

get_weights_sql = """
select * from (
    select w, w0 from cluster_0 where parent = ? union
    select w, w0 from cluster_1 where parent = ? union
    select w, w0 from cluster_2 where parent = ? union
    select w, w0 from cluster_3 where parent = ? union
    select w, w0 from cluster_4 where parent = ? union
    select w, w0 from cluster_5 where parent = ?
)
"""

connection = sqlite3.connect("experiments.sqlite")
combinations = [('ball', 1), ('cube', 1), ('simplex', 1), ('ball', 2), ('cube', 2), ('simplex', 2)]


def standard_scaler(name: str, n: int, k: int, seed: int) -> StandardScaler:
    df = DataModel(name, k=k, n=n, seed=seed).train_set()
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler


def destandardize(scaler: StandardScaler, W: np.array, nc: int) -> np.array:
    W[0] /= np.tile(scaler.scale_, nc)
    W[1] += np.sum(np.split(W[0] * np.tile(scaler.mean_, nc), nc), axis=1)
    return W


def to_mathematica(W, nc):
    output = ""
    for i, c in enumerate(np.split(W[0], nc)):
        for j in range(c.shape[0]):
            output += "%fx[%d] + " % (c[j], j+1)
        output = output[:-2] + "< %f &&\n" % W[1][i]
    output = output[:-4]
    return output


if __name__ == '__main__':
    for params in combinations:
        model = f'{params[0]}_{params[1]}_3'
        data = []
        res = connection.execute(get_params_sql, params).fetchone()
        nc = n_constraints(res[1], 3)
        scaler = standard_scaler(params[0], 3, params[1], res[2]) if res[3] else None
        constraints = connection.execute(get_weights_sql, [res[0]] * 6).fetchall()

        w0 = constraints[0][1].split(' ')
        w0 = list(map(lambda x: float(x), w0))

        w = constraints[0][0].split(' ')
        w = list(map(lambda x: float(x), w))
        w = np.array(w)
        
        constraints = [w, w0]
        if scaler is not None:
            constraints = destandardize(scaler, constraints, nc)
        text = to_mathematica(constraints, nc)
        log.debug(f"Best model for: {model}:\n{text}")
        write_file(model, text, '.txt', Paths.best_models.value)

