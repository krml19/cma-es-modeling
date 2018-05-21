import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler
# from scripts.utils.latexify import DataTable


class Aggragator:
    attribute: str = 'standardized'
    attribute_values: list = None
    experiment: int = 1

    def fit(self):
        sql = "SELECT * FROM experiments WHERE benchmark_mode=0 AND experiment_n={}".format(self.experiment)
        conn = sqlite3.connect("experiments.sqlite")
        # sql = "SELECT * FROM experiments"
        # conn = sqlite3.connect("benchmarks.sqlite")
        df = pd.read_sql_query(sql, conn)
        conn.close()

        grouping_attributes = ['n_constraints', 'clustering', 'margin', 'standardized', 'sigma', 'name', 'k', 'n']

        df2 = df.groupby(grouping_attributes).apply(self.get_stats)
        data_frame = self.expand_dataframes(df2, self.attribute)
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

    def normalize(self, df):
        scaler = MinMaxScaler()
        scaler.fit(df.values.reshape(-1, 1))
        df = df.applymap(lambda x: scaler.transform(x)[0][0])
        return df

    def expand_dataframes(self, df2, ranking_attribute: str):
        data_frame: pd.DataFrame = df2.groupby(['name', ranking_attribute])[['f_mean', 'f_sem']].mean().unstack()
        self.attribute_values = list(df2[self.attribute].unique())

        rank_keys = [('rank', key) for key in self.attribute_values]
        data_frame[rank_keys] = data_frame['f_mean'].rank(axis=1, ascending=False)

        rank_norm_keys = [('rank_norm', key) for key in self.attribute_values]
        data_frame[rank_norm_keys] = self.normalize(data_frame['rank'])

        return data_frame


# aggregator = Aggragator()
# data_frame = aggregator.fit()
#
# table = DataTable(data_frame, 'Standaryzacja', attribute=aggregator.attribute, attribute_values=aggregator.attribute_values)
# l = table.table()
# print(l)
