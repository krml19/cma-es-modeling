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

