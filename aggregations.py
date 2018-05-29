import pandas as pd
import sqlite3
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import scipy.stats as stats
import numpy as np

class Aggragator:
    def __init__(self, experiment: int, attribute: str, benchmark_mode: bool = False):

        self.attribute = attribute
        self.experiment = experiment
        self.attribute_values: list = None
        self.benchmark_mode = int(benchmark_mode)
        self.info = None

    def transform(self) -> pd.DataFrame:
        sql = "SELECT * FROM experiments WHERE benchmark_mode={} AND experiment_n={}".format(self.benchmark_mode, self.experiment)
        conn = sqlite3.connect("experiments.sqlite")
        df = pd.read_sql_query(sql, conn)
        conn.close()

        grouping_attributes = ['constraints_generator', 'clustering', 'margin', 'standardized', 'sigma', 'name', 'k', 'n']
        self.get_info(df)
        df2 = df.groupby(grouping_attributes).apply(self.__get_stats)
        data_frame = self.__expand_dataframes(df2, self.attribute)
        return data_frame

    def get_info(self, df: pd.DataFrame):
        info = dict()

        def reducer(X):
            return str(set(X.unique())).replace('\'', '')

        info['margin'] = df['margin'].iloc[0]
        info['n'] = "{} - {}".format(df['n'].min(), df['n'].max())
        info['k'] = "{} - {}".format(df['k'].min(), df['k'].max())
        info['margin'] = df['margin'].iloc[0]
        info['sigma'] = df['sigma'].iloc[0]
        info['standardized'] = reducer(df['standardized'])
        info['constraints_generator'] = reducer(df['constraints_generator'])
        info['clustering'] = reducer(df['clustering'])
        info[self.attribute] = reducer(df[self.attribute])
        info['seed'] = reducer(df['seed'])
        info['total_experiments'] = df.shape[0]

        self.info = info

    def __get_stats(self, group):
        results = {
            'f_mean': group['f'].mean() * -1,
            'f_sem': group['f'].sem(ddof=1) * stats.norm.ppf(q=0.975),
            'tp_mean': group['tp'].mean(),
            'tn_mean': group['tn'].mean(),
            'fp_mean': group['fp'].mean(),
            'fn_mean': group['fn'].mean(),
            'standardized': group['standardized'].iloc[0],
            'n_constraints': group['n_constraints'].iloc[0],
            'constraints_generator': group['constraints_generator'].iloc[0],
            'clustering': group['clustering'].iloc[0],
            'margin': group['margin'].iloc[0],
            'sigma': group['sigma'].iloc[0],
            'model': "{}_{}_{}".format(group['name'].iloc[0], group['k'].iloc[0], group['n'].iloc[0]),
            'name': group['name'].iloc[0],
            'n': group['n'].iloc[0],
            'k': group['k'].iloc[0],
        }
        return pd.Series(results, name='metrics')

    def __normalize(self, series: pd.Series):
        # series = series.applymap(lambda x: np.log(1+x))
        scaler = QuantileTransformer()
        scaler.fit(series.values.reshape(-1, 1))
        series = series.applymap(lambda x: scaler.transform(x)[0][0])
        return series

    def __expand_dataframes(self, df2, ranking_attribute: str):
        data_frame: pd.DataFrame = df2.groupby(['model', ranking_attribute])[['f_mean', 'f_sem']].mean().unstack()
        self.attribute_values = list(df2[self.attribute].unique())

        rank_keys = [('rank', key) for key in self.attribute_values]
        data_frame[rank_keys] = data_frame['f_mean'].rank(axis=1, ascending=False)

        rank_norm_keys = [('rank_norm', key) for key in self.attribute_values]
        data_frame[rank_norm_keys] = self.__normalize(data_frame['f_mean'])

        return data_frame




# aggregator = Aggragator()
# data_frame = aggregator.fit()
#
# table = DataTable(data_frame, 'Standaryzacja', attribute=aggregator.attribute, attribute_values=aggregator.attribute_values)
# l = table.table()
# print(l)
