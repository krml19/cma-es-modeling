import numpy as np


def f_2n(n: int) -> int:
    return 2 * n

def f_2np2(n: int) -> int:
    return 2 * n ** 2


def f_n3(n: int) -> int:
    return int(np.power(n, 3))


def f_2pn(n: int) -> int:
    return int(np.power(2, n))


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
    functions = [f_2n, f_2np2, f_2pn, f_n3]

    x = np.arange(2, 12)

    items = dict()
    for f in functions:
        items[to_latex(f.__name__)] = list(map(lambda xi: f(xi), x))

    df = pd.DataFrame.from_dict(items)
    df.set_index(x, inplace=True)
    df.plot(xticks=x, xlim=(2, x.max()))
    plt.show()
    return df


if __name__ == '__main__':
    df = draw()
