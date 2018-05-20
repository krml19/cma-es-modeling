import pandas as pd
import sqlite3
import scipy.stats as stats
import numpy as np
import itertools


class Ranker:
    def __init__(self):
        self.values: dict = None

    def fit(self, X: [list, np.array]):
        v = dict()
        i = 1
        X.sort()
        for key, group in itertools.groupby(X):
            _list = list(group)
            _len = len(_list)
            _sum = np.arange(start=i, stop=i + _len, step=1).sum()
            v[key] = (_sum / _len)
            i = i + _len
        self.values = v

    def predict(self, X: [list, float]):
        assert self.values is not None
        if isinstance(X, float):
            return self.values[X]
        return list(map(lambda x: self.values[x], X))


class Aggragator:
    def __init__(self):
        sql = "select * from experiments"
        conn = sqlite3.connect("experiments.sqlite")
        df = pd.read_sql_query(sql, conn)
        conn.close()

        grouping_attributes = ['n_constraints', 'clustering', 'margin', 'standardized', 'sigma', 'name', 'k', 'n']

        df2 = df.groupby(grouping_attributes).apply(self.get_stats)
        # df2['rank'] = self.rank(df2)
        df2['rank'] = self.groups(df2, grouping_attributes, 'standardized')

        print(df2)

    def get_stats(self, group):
        results = {
            'f_mean': group['f'].mean(),
            'f_sem': group['f'].sem(),
            'tp_mean': group['tp'].mean(),
            'tn_mean': group['tn'].mean(),
            'fp_mean': group['fp'].mean(),
            'fn_mean': group['fn'].mean(),
            'standardized': group['standardized'].iloc[0],
            'n_constraints': group['n_constraints'].iloc[0],
            'clustering': group['clustering'].iloc[0],
            'margin': group['margin'].iloc[0],
            'sigma': group['sigma'].iloc[0],
            'name': group['name'].iloc[0],
            'k': group['k'].iloc[0],
            'n': group['n'].iloc[0]
        }
        return pd.Series(results, name='metrics')

    def rank(self, group: pd.Series) -> pd.Series:
        ranker = Ranker()
        ranker.fit(group.values)
        return group.apply(ranker.predict)

    def groups(self, df2, grouping_attributes, rank_attribute):
        grouping_attributes = set(grouping_attributes)
        rank_attribute = set(rank_attribute)
        grouping_attributes = list(grouping_attributes.difference(rank_attribute))
        groups = df2.groupby(grouping_attributes)
        return groups['f_mean'].apply(self.rank)





Aggragator()

