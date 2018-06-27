import numpy as np


def f_2n(n):
    return 2 * n

def f_2np2(n):
    return 2 * n ** 2


def f_n3(n):
    return np.power(n, 3)


def f_2pn(n):
    return np.power(2, n)


def generate(f_name, value):
    return {f_2n.__name__: f_2n(value),
            f_2np2.__name__: f_2np2(value),
            f_n3.__name__: f_n3(value),
            f_2pn.__name__: f_2pn(value)}[f_name]


def to_latex(f_name):
    return {f_2n.__name__: r'$2n$',
            f_2np2.__name__: r'$2n^2$',
            f_n3.__name__: r'$n^3$',
            f_2pn.__name__: r'$2^n$'}[f_name]


def draw():
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib2tikz import save as tikz_save
    functions = [f_2n, f_2np2, f_2pn, f_n3]

    xmin = 2
    xmax = 7
    x = np.arange(xmin, xmax, 0.1)
    ticks = np.arange(xmin, xmax)
    items = dict()
    for f in functions:
        items[to_latex(f.__name__)] = list(map(lambda xi: f(xi), x))

    df = pd.DataFrame.from_dict(items)
    df.set_index(x, inplace=True)
    ax = df.plot(xticks=ticks, xlim=(2, x.max()), logy=True)

    ax.set_xlabel('$n$')
    ax.set_ylabel('$n_c$')
    # plt.show()
    tikz_save("cg-%d-%d.tex" % (xmin, xmax))
    return df


if __name__ == '__main__':
    df = draw()
