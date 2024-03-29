import pandas as pd
import sqlite3
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import scipy.stats as stats
import runner
import numpy as np
from logger import Logger
import time
import itertools
from functools import reduce
import math

log = Logger(name='cma-es')


def identity(x):
    print(x)
    return x


class Measure:
    grouping_attribute: str = None
    sem: str = None
    name: str = None

class MeasureF(Measure):
    grouping_attribute = 'f_mean'
    sem = 'f_sem'
    name = 'f'

class MeasureF1(Measure):
    grouping_attribute = 'f1_mean'
    sem = 'f1_sem'
    name = 'F_1'

class MeasureTime(Measure):
    grouping_attribute = 'time_mean'
    sem = 'time_sem'
    name = 'CPU Time'


class Aggragator:
    def __init__(self, experiment, attribute: str = None, benchmark_mode: bool = False, measures: [Measure] = [MeasureF]):

        self.attribute = attribute
        self.experiment = experiment
        self.attribute_values: list = None
        self.benchmark_mode = int(benchmark_mode)
        self.measures: [Measure] = measures
        self.info = None
        self.cm = None

    def db(self):
        algorithm_runner = runner.AlgorithmRunner()
        sql = "SELECT * FROM experiments WHERE constraints_generator=? AND margin=? AND sigma=? AND k=? AND n=? AND seed=? AND name=? AND clustering=? AND standardized=? AND (train_tp + train_fn)=?"
        connection = sqlite3.connect("experiments.sqlite")
        queries = runner.flat(algorithm_runner.experiment(self.experiment, seeds=range(0, 30)))
        df = pd.concat(
            [pd.read_sql_query(sql=sql, params=algorithm_runner.convert_to_sql_params(q), con=connection) for q in
             queries])

        connection.close()
        df['train_sample'] = df['train_tp'] + df['train_fn']
        df['f1'] = self.f1_score(df)

        return df

    def transform(self, split: [None, list] = None, rank_ascending=False) -> pd.DataFrame:
        db: pd.DataFrame = self.db()
        grouping_attributes = ['constraints_generator', 'clustering', 'margin', 'standardized', 'sigma', 'name', 'k',
                               'n', 'train_sample']

        if split is not None:
            unique = [db[key].unique() for key in split]
            combinations = list(itertools.product(*unique))
            items = list()
            for combination in combinations:
                query = reduce(lambda x, y: x + ' & ' + y,
                               map(lambda x: "{} == {}".format(x[0], '\'%s\'' % x[1] if isinstance(x[1], str) else x[1]), zip(split, list(combination))))
                chunk = db.query(query)
                df2 = chunk.groupby(grouping_attributes).apply(self.__get_stats)
                data_frame = self.select_features(df2, self.attribute, rank_ascending=rank_ascending)
                items.append((data_frame, combination))
            return items
        else:
            df2 = db.groupby(grouping_attributes).apply(self.__get_stats)
            data_frames = [self.select_features(df2, self.attribute, measure=measure, rank_ascending=rank_ascending) for measure in self.measures]
            self.update_info(db, df2)
            return pd.concat(data_frames, keys=map(lambda measure: measure.name, self.measures))

    @staticmethod
    def f1_score(df: [pd.DataFrame, pd.Series]) -> pd.Series:
        # print(df[["tp", "fp", "tn", "fn"]])
        p = df.tp / (df.tp + df.fp + 1e-12)  # tp + fp = 0 -> p = 0
        r = df.tp / (df.tp + df.fn + 1e-12)  # tp + fn = 0 -> r = 0
        return 2.0 * p * r / (p + r + 1e-12)

    @staticmethod
    def mcc(df: [pd.DataFrame, pd.Series]) -> pd.Series:
        return (df.tp * df.tn - df.fp * df.fn) / ((df.tp + df.fp) * (df.tp + df.fn) * (df.tn + df.fp) * (df.tn + df.fn)).apply(lambda x: max(math.sqrt(x), 1))

    @staticmethod
    def accuracy(df: [pd.DataFrame, pd.Series]) -> pd.Series:
        return (df.tp + df.tn) / (df.tp + df.tn + df.fp + df.fn)

    @staticmethod
    def recall(df: [pd.DataFrame, pd.Series]) -> pd.Series:
        return df.tp / (df.tp + df.fn)

    @staticmethod
    def precision(df: [pd.DataFrame, pd.Series]) -> pd.Series:
        return df.tp / (df.tp + df.fp).apply(lambda x: max(x, 1))

    def confusion_matrix(self):
        df = self.db()
        grouping_attributes = ['name', 'k']
        cm: pd.DataFrame = df.groupby(grouping_attributes).apply(self.cm_stats)
        cm = cm.apply(self.map_series)
        return cm

    def map_series(self, series: pd.Series) -> pd.Series:
        print(series)
        m = series.apply(lambda x: x[1]).max()
        return series.apply(lambda x: (x[0], x[1] / m))

    @staticmethod
    def objective_function(df: [pd.DataFrame, pd.Series]) -> pd.Series:
        return df.f * -1

    def cm_stats(self, group):
        func = lambda x: (x(group).mean(), x(group).sem(ddof=1) * stats.t.interval(alpha=0.95, df=group.shape[0])[1])
        f1 = func(self.objective_function)
        results = {
            'ACC': func(self.accuracy),
            'p': func(self.precision),
            'r': func(self.recall),
            'F_1': func(self.f1_score),
            'MCC': func(self.mcc),
            'f': func(self.objective_function)
            # 'f': (f1[0], f1[1]) # / f1[0]),
        }

        return pd.Series(results, name='metrics')

    def update_info(self, df: pd.DataFrame, df2: pd.DataFrame):
        info = dict()

        def reducer(X):
            return str(set(X.unique())).replace('\'', '')

        info['date'] = time.asctime()
        info['margin'] = df['margin'].iloc[0]
        info['n'] = "{} - {}".format(df['n'].min(), df['n'].max())
        info['k'] = "{} - {}".format(df['k'].min(), df['k'].max())
        info['margin'] = df['margin'].iloc[0]
        info['sigma'] = df['sigma'].iloc[0]
        info['s'] = reducer(df['standardized'])
        info['cq'] = reducer(df['constraints_generator'])
        info['clustering'] = reducer(df['clustering'])
        info['seed'] = reducer(df['seed'])
        info['total'] = df.shape[0]

        info[self.attribute] = reducer(df[self.attribute])
        info[('model', 'attribute')] = 'len(seeds)'
        for key, value, attribute in zip(df2['model'], df2['seeds'], df2[self.attribute]):
            info[(key, attribute)] = value

        self.info = info

    def __get_stats(self, group):
        results = {
            'f_mean': (group['f'].mean() * -1),
            'f1_mean': (group['f1'].mean()),
            'time_mean': (group['time'].mean()),
            'f_sem': (group['f'].sem(ddof=1) * stats.t.interval(alpha=0.95, df=group.shape[0])[1]), # / (group['f'].mean() * -1),
            'f1_sem': (group['f1'].sem(ddof=1) * stats.t.interval(alpha=0.95, df=group.shape[0])[1]),
            'time_sem': (group['time'].sem(ddof=1) * stats.t.interval(alpha=0.95, df=group.shape[0])[1]),
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
            'train_sample': group['train_sample'].iloc[0],
            'seeds': group['seed'].nunique()
        }
        return pd.Series(results, name='metrics')

    def normalize(self, series: pd.Series):
        series = series.fillna(0)
        scaler = QuantileTransformer()
        scaler.fit(series.values.ravel().reshape(-1, 1))

        for col in range(series.values.shape[1]):
            series.values[:, col] = scaler.transform(series.values[:, col].reshape(-1, 1)).ravel()

        return series

    def normalize_sem(self, series: pd.Series, measure: Measure):
        # if measure is MeasureF:
        #     max = 1
        # else:
        #     max  = 1
        # scaler = MinMaxScaler(feature_range=(0, max))
        series = series.fillna(0)
        # scaler.fit(series.values.reshape(-1, 1))
        # series = series.applymap(lambda x: scaler.transform(x)[0][0])
        return series

    def select_features(self, df2, ranking_attribute: str, measure: Measure = MeasureF, rank_ascending=False):
        data_frame: pd.DataFrame = df2.groupby(['model', ranking_attribute])[[measure.grouping_attribute, measure.sem]].mean().unstack()
        self.attribute_values = list(df2[self.attribute].unique())

        rank_keys = [('rank', key) for key in self.attribute_values]
        data_frame[rank_keys] = data_frame[measure.grouping_attribute].rank(axis=1, ascending=rank_ascending)

        rank_norm_keys = [('rank_norm', key) for key in self.attribute_values]
        data_frame[rank_norm_keys] = self.normalize(data_frame[measure.grouping_attribute])

        sem_norm_keys = [('sem_norm', key) for key in self.attribute_values]
        data_frame[sem_norm_keys] = self.normalize_sem(data_frame[measure.sem], measure=measure)

        data_frame.rename(columns={
            measure.grouping_attribute: MeasureF.grouping_attribute,
            measure.sem: MeasureF.sem
        }, inplace=True)

        return data_frame
