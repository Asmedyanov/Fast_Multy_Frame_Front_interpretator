import numpy as np


def f_line(x, a, b):
    return a * x + b


def f_line_shelf(x, t0, b):
    a = -b / t0
    ret = np.where(x < t0, 0, a * x + b)
    return ret


def f_bipower(t, t0, a0, b0, d, a1, power1, power2):
    a2 = (a0 - a1 * power1 * np.power(t0, power1 - 1.0)) * np.power(t0, 1.0 - power2) / power2
    h = a0 * t0 + b0 + d - a1 * np.power(t0, power1) - a2 * np.power(t0, power2)
    y = np.where(t > t0, a0 * t + b0 + d,
                 a1 * np.power(t, power1) + a2 * np.power(t, power2) + h)
    return y


def f_square_line(t, t0, a0, b0, d0, t1):
    a = a0 / (2.0 * (t0 - t1))
    d = a0 * t0 + b0 + d0 - a * np.square(t0 - t1)
    y = np.where(t < t0,
                 a * np.square(t - t1) + d,
                 a0 * t + b0 + d0)
    return y
