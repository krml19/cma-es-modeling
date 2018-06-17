import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


class Probabilities:

    def transform(self):
        x = np.arange(1, 10)
        data = [math.pow(0.5, n) for n in x]

        df = pd.DataFrame(data=data, columns=['$Objętość\, obszaru\, dopuszczalnego$'])
        df.set_index(x, inplace=True)

        ax = df.plot(logy=True, figsize=(8, 4))

        ax.set_xlabel('$n$')
        ax.set_ylabel('$V$')

        plt.show()


p = Probabilities()
p.transform()

