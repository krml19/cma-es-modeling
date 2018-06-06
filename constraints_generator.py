import numpy as np


def f_2n(n: int) -> int:
    return 2 * n


def f_2n2(n: int) -> int:
    return 2 * n ** 2


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
