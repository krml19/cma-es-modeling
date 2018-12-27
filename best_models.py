import sqlite3
from functools import reduce

import numpy as np
from file_helper import write_tex_table, Paths

from constraints_generator import generate as n_constraints
sql = """
select id, constraints_generator from experiments where name = ? and n=3 and k=? order by f limit 1
"""

sql2 = """
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

# params = combinations[0]


if __name__ == '__main__':
    for params in combinations:
        model = f'{params[0]}_{params[1]}_3'
        data = []
        res = connection.execute(sql, params).fetchone()
        nc = n_constraints(res[1], 3)
        constraints = connection.execute(sql2, [res[0]] * 6).fetchall()

        w0 = list(map(lambda x: x[1].split(' '), constraints))
        w0 = list(map(lambda x: np.reshape(x, (nc, 1)).tolist(), w0))

        constraints = list(map(lambda x: x[0].split(' '), constraints))
        constraints = list(map(lambda x: np.reshape(x, (nc, 3)).tolist(), constraints))

        sign = lambda x: f"+ {x}" if float(x) > 0 else f"- {-1 * float(x)}"
        for i, (c_cluster, w_cluster) in enumerate(zip(constraints, w0)):
            print(f"Cluster {i}:")
            for index, (c, w) in enumerate(zip(c_cluster, w_cluster)):
                constraint = f'c{index}: {c[0]}x1 {sign(c[1])}x2 + {sign(c[2])}x3 < {w[0]}'
                data.append(constraint)
        write_tex_table(model, reduce(lambda x, y: x + '\n' + y, data), '.txt', Paths.best_models.value)
        # subject_to = f"""
        # Subject To
        # {constraints}
        # """

