import pandas as pd
import sqlite3
from sklearn.preprocessing import QuantileTransformer
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
    name = 'f1'


class Aggragator:
    def __init__(self, experiment, attribute: str = None, benchmark_mode: bool = False, measures: [Measure] = [MeasureF, MeasureF1]):

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

    def transform(self, split: [None, list] = None) -> pd.DataFrame:
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
                data_frame = self.select_features(df2, self.attribute)
                items.append((data_frame, combination))
            return items
        else:
            df2 = db.groupby(grouping_attributes).apply(self.__get_stats)
            data_frames = [self.select_features(df2, self.attribute, measure=measure) for measure in self.measures]
            self.update_info(db, df2)
            return pd.concat(data_frames, keys=map(lambda measure: measure.name, self.measures))

    @staticmethod
    def f1_score(df: pd.DataFrame) -> pd.Series:
        return 2 * df.tp / (2 * df.tp + df.fp + df.fn)

    @staticmethod
    def mcc(df: pd.DataFrame) -> pd.Series:
        return (df.tp * df.tn - df.fp * df.fn) / ((df.tp + df.fp) * (df.tp + df.fn) * (df.tn + df.fp) * (df.tn + df.fn)).apply(lambda x: math.sqrt(x))

    @staticmethod
    def accuracy(df: pd.DataFrame) -> pd.Series:
        return (df.tp + df.tn) / (df.tp + df.tn + df.fp + df.fn)

    @staticmethod
    def recall(df: pd.DataFrame) -> pd.Series:
        return df.tp / (df.tp + df.fn)

    @staticmethod
    def precision(df: pd.DataFrame) -> pd.Series:
        return df.tp / (df.tp + df.fp)

    def confusion_matrix(self):
        df = self.db()
        cm = pd.DataFrame()

        cm['ACC'] = self.accuracy(df)
        cm['p'] = self.precision(df)
        cm['r'] = self.recall(df)
        cm['F_1'] = self.f1_score(df)
        cm['MCC'] = self.mcc(df)

        cm2 = dict()
        for key in cm.keys():
            cm2[key] = "%0.3f" % cm[key].mean()
        return cm2

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
            'f_sem': group['f'].sem(ddof=1) * stats.norm.ppf(q=0.975),
            'f1_sem': group['f1'].sem(ddof=1) * stats.norm.ppf(q=0.975),
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

    def __normalize(self, series: pd.Series):
        scaler = QuantileTransformer()
        series = series.fillna(0)
        scaler.fit(series.values.reshape(-1, 1))
        series = series.applymap(lambda x: scaler.transform(x)[0][0])
        return series

    def select_features(self, df2, ranking_attribute: str, measure: Measure = MeasureF):
        data_frame: pd.DataFrame = df2.groupby(['model', ranking_attribute])[[measure.grouping_attribute, measure.sem]].mean().unstack()
        self.attribute_values = list(df2[self.attribute].unique())

        rank_keys = [('rank', key) for key in self.attribute_values]
        data_frame[rank_keys] = data_frame[measure.grouping_attribute].rank(axis=1, ascending=False)

        rank_norm_keys = [('rank_norm', key) for key in self.attribute_values]
        data_frame[rank_norm_keys] = self.__normalize(data_frame[measure.grouping_attribute])

        sem_norm_keys = [('sem_norm', key) for key in self.attribute_values]
        data_frame[sem_norm_keys] = self.__normalize(data_frame[measure.sem])

        data_frame.rename(columns={
            measure.grouping_attribute: MeasureF.grouping_attribute,
            measure.sem: MeasureF.sem
        }, inplace=True)

        return data_frame
