import pandas as pd
import sqlite3
import scipy.stats as stats
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from scripts.utils.latexify import Table

class Ranker:
    def __init__(self):
        self.values: dict = None

    def fit(self, X: [list, np.array]):
        v = dict()
        i = 1
        X = -np.sort(-X)
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
    def fit(self):
        sql = "SELECT * FROM experiments"
        conn = sqlite3.connect("experiments.sqlite")
        df = pd.read_sql_query(sql, conn)
        conn.close()

        grouping_attributes = ['n_constraints', 'clustering', 'margin', 'standardized', 'sigma', 'name', 'k', 'n']

        df2 = df.groupby(grouping_attributes).apply(self.get_stats)
        # df2['rank'] = self.rank(df2)
        # df2['rank'] = self.groups(df2, grouping_attributes, 'standardized')
        data_frame = self.expand_dataframes(df2, 'standardized')
        return data_frame


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
            'name': "{}_{}_{}".format(group['name'].iloc[0], group['n'].iloc[0], group['k'].iloc[0]),
        }
        return pd.Series(results, name='metrics')

    def rank(self, group: pd.Series) -> pd.Series:
        ranker = Ranker()
        ranker.fit(group.values)
        return group.apply(ranker.predict)

    def apply_rank(self, X):
        ranker = Ranker()
        ranker.fit(X.values)
        return ranker.predict(X.values)

    def normalize(self, df, prefix='norm_'):
        scaler = MinMaxScaler()
        scaler.fit(df.values.reshape(-1, 1))
        df = df.applymap(lambda x: scaler.transform(x)[0][0])
        df = df.add_prefix(prefix=prefix)
        return df

    def groups(self, df2, grouping_attributes, rank_attribute):
        grouping_attributes = set(grouping_attributes)
        rank_attribute = set(rank_attribute)
        grouping_attributes = list(grouping_attributes.difference(rank_attribute))
        groups = df2.groupby(grouping_attributes)
        return groups['f_mean'].apply(self.rank)

    def expand_dataframes(self, df2, ranking_attribute: str):
        data_frame: pd.DataFrame = df2.groupby(['name', ranking_attribute])['f_mean'].mean().unstack()
        ranking = data_frame.apply(self.apply_rank, axis=1)
        ranking: pd.DataFrame = ranking.add_prefix('rank_')
        normalized_rank = self.normalize(ranking)
        df = data_frame.join(ranking).join(normalized_rank)
        return df


data_frame = Aggragator().fit()
table = Table(data_frame, 'Standaryzacja')
l = table.table()
print(l)
