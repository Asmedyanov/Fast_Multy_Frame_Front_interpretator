import numpy as np
from numpy import where


def f_line(x, a, b):
    return a * x + b


def f_line_shelf(x, t0, b):
    a = -b / t0
    ret = np.where(x < t0, 0, a * x + b)
    return ret


def f_power_shelf(x, t0, a, b):
    ret = np.where(x < t0, 0, a * np.power(x - t0, b))
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


def f_square_line_time(t, t0, t1, a):
    b = -2 * a * t1
    c = -a * t0 ** 2 + 2 * a * t0 * t1
    ret = np.where(t < t0, 0, a * t ** 2 + b * t + c)
    return ret


def f_square_line_time_reversed(x, a, c):
    ret = a * x + c
    return ret


def f_hard_core(x, a0, b0, d0, x0_s, x1_s, x0_l, x1_l, x0_v, x1_v, x0_p, x1_p):
    a_s = a0 / (2.0 * (x1_s - x0_s))
    c_s = a0 * x1_s + b0 + d0 - a_s * (x1_s - x0_s) ** 2
    a_l = a_s * (x1_l - x0_s) / (x1_l - x0_l)
    c_l = a_s * (x1_l - x0_s) ** 2.0 + c_s - a_l * (x1_l - x0_l) ** 2.0
    a_v = a_l * (x1_v - x0_l) / (x1_v - x0_v)
    c_v = a_l * (x1_v - x0_l) ** 2.0 + c_l - a_v * (x1_v - x0_v) ** 2.0
    a_p = a_l * (x1_p - x0_v) / (x1_p - x0_p)
    c_p = a_l * (x1_p - x0_v) ** 2.0 + c_v - a_p * (x1_p - x0_p) ** 2.0
    ret = where(
        x<x1_p,
        a_p*(x-x0_p)**2.0+c_p,
        where(
            x<x1_v,
            a_v * (x - x0_v) ** 2.0 + c_v,
            where(
                x < x1_l,
                a_l * (x - x0_l) ** 2.0 + c_l,
                where(
                    x < x1_s,
                    a_s * (x - x0_s) ** 2.0 + c_s,
                    a0*x+b0+d0
                )
            )
        )

    )
    return ret
