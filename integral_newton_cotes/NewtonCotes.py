from func import *
import numpy as np
import errors as err
import math


def iqf(a, b):
    n = 3
    # узлы
    x_j = [a, (a + b) / 2, b]
    mu_j = np.array([mu_i(a, b, i) for i in range(n)], dtype=float)
    # решаем систему sigma A_j*x_js = 0 , j=0,1,2
    X_js = np.array([[(x_j[j]) ** i for j in range(n)]for i in range(n)])
    A_i = np.linalg.solve(X_js, mu_j)
    result = 0
    for i in range(n):
        result += A_i[i] * f(X_js[1][i])
    return result


def complexIqf(a, b, steps=0, delta=0):
    result = 0
    if steps != 0:
        h = (b - a) / steps
        t1 = a
        t2 = a + h
        for _ in range(steps):
            result += iqf(t1, t2)
            t1 = t2
            t2 += h
        return result
    if delta != 0:
        t1 = a
        t2 = a + delta
        while t2 < b:
            result += iqf(t1, t2)
            t1 = t2
            t2 += delta
        result += iqf(t1, b)
        return result


def complexIqf_err(a, b, eps=1e-6):
    S_h = []
    steps = 2
    for i in range(0, steps):
        S_h.append(complexIqf(a, b, steps=2 ** i))
    R_h = 1
    while abs(R_h) > eps:
        S_h.append(complexIqf(a, b, 2 ** steps))
        sol, m = err.Richardson(S_h[-3:], b - a, L = 2)
        R_h = abs(sol - S_h[-1])
        print(f"h={(b-a)/2**steps}, погрешность: {R_h}, скорость сходимости по Эйткинсу: {m}")
        steps += 1


def complexIqf_opt(a, b, eps=1e-6):
    L = 2
    steps = 3
    S_h = [complexIqf(a, b, steps=2 ** i) for i in range(0, steps)]
    sol, m = err.Richardson(S_h, b - a, L)
    h_opt = (b-a) / 4 * (eps * (L ** (m) - 1) / abs(S_h[-1] - S_h[-2])) ** (1 / m)
    h_opt = (b - a) / math.ceil((b - a) / h_opt)
    h_init = h_opt
    S_h = [complexIqf(a, b, delta=h_opt), complexIqf(a, b, delta=h_opt / L)]
    h_opt /= (L ** 2)
    R_h = 1
    while abs(R_h) > eps:
        S_h.append(complexIqf(a, b, delta=h_opt))
        sol, m = err.Richardson(S_h, h_opt, L=L)
        R_h = abs(sol - S_h[-1])
        print(f"h={h_opt}, погрешность: {R_h}, скорость сходимости по Эйткинсу: {m}")
        steps += 1
        h_opt /= L


