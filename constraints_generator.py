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


def draw(path='.'):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib2tikz import save as tikz_save
    functions = [f_2n, f_2pn, f_2np2, f_n3]

    xmin = 2
    xmax = 12
    x = np.arange(xmin, xmax, 1)
    ticks = np.arange(xmin, xmax)
    columns = [to_latex(f.__name__) for f in functions]
    data = []

    for xi in x:
        data.append(list(map(lambda f: f(xi), functions)))


    df = pd.DataFrame(data=data, columns=columns)

    df.set_index(x, inplace=True)
    ax = df.plot.bar(xticks=ticks, logy=False)

    ax.set_xlabel('$n$')
    ax.set_ylabel('$n_c$')
    # plt.show()
    # tikz_save("%s/cg-%d-%d.tex" % (path, xmin, xmax-1))
    plt.savefig("%s/cg-%d-%d.pdf" % (path, xmin, xmax-1))
    return df


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        draw(path=sys.argv[1])
