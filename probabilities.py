import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Probabilities:
    sql = "SELECT * FROM experiments"
    db = 'experiments.sqlite'
    df: pd.DataFrame = None
    grouping_attributes = ['n', 'k', 'problem']
    grouped: pd.DataFrame = None

    def fit(self):
        connection = sqlite3.connect(self.db)
        self.df = pd.read_sql_query(sql=self.sql, con=connection)
        print("Number of rows: %i" % self.df.shape[0])
        self.df.rename(columns={'name': 'problem'}, inplace=True)
        connection.close()

    def transform(self):
        self.grouped = self.df.groupby(self.grouping_attributes).apply(self.__get_stats)
        data = self.grouped.unstack().unstack().values[:, 0:4].mean(axis=1)
        df = pd.DataFrame(data=data, columns=['$Problem$'])
        # for y in data:
        #     plt.plot()
        # # self.grouped.plot()
        x = np.arange(2, 8)
        df.set_index(x, inplace=True)

        ax = df.plot(logy=True)

        ax.set_xlabel('$n$')
        ax.set_ylabel('$Pr[Y=1]$')


        plt.show()

    def __get_stats(self, group):
        results = {
            'probability of positive class': (group['tp'].mean()) / 1e6,
        }
        return pd.Series(results, name='probability of positive class')


p = Probabilities()
p.fit()
p.transform()

