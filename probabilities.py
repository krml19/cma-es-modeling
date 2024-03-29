import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
# from matplotlib2tikz import save as tikz_save
# from matplotlib2tikz import get_tikz_code


class Probabilities:

    def transform(self, path='.'):
        x = np.arange(1, 10)
        data = [math.pow(0.5, n) for n in x]

        df = pd.DataFrame(data=data, columns=['$Objętość\, obszaru\, dopuszczalnego$'])
        df.set_index(x, inplace=True)

        ax = df.plot.bar(logy=True, figsize=(8, 4))

        ax.set_xlabel('$n$')
        ax.set_ylabel('$V$')
        ax.legend_.remove()
        # tikz_save("%s/pryn.tex" % path)
        # print(get_tikz_code())
        # plt.show()
        plt.savefig("%s/pryn.pdf" % path)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        p = Probabilities()
        p.transform(sys.argv[1])



# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.style.use("ggplot")
#
# t = np.arange(0.0, 2.0, 0.1)
# s = np.sin(2 * np.pi * t)
# s2 = np.cos(2 * np.pi * t)
# plt.plot(t, s, "o-", lw=4.1)
# plt.plot(t, s2, "o-", lw=4.1)
# plt.xlabel("time (s)")
# plt.ylabel("Voltage (mV)")
# plt.title("Simple plot $\\frac{\\alpha}{2}$")
# plt.grid(True)
#

