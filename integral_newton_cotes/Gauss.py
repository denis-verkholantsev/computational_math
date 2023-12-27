import numpy as np
import errors as err
from func import *
import math


def iqf(a, b):
    size = 3
    mu_j = np.array([mu_i(a, b, i) for i in range(2*size)], dtype=float)
    X_js = np.array([[mu_j[i+j] for j in range(size)]for i in range(size)])

    b = -mu_j[size:]
    A = np.linalg.solve(X_js, b)

    A = (np.append(A, 1))[::-1]
    solution = np.roots(A)

    for i in range(size):
        X_js[i][:] = solution ** i

    Amat = np.linalg.solve(X_js, mu_j[0:size])
    result = 0
    for i in range(size):
        result += Amat[i] * f(X_js[1][i])

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
    steps = 3
    for i in range(1, steps):
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
    h_opt = (b - a) * (eps * (1 - L ** (-m)) / abs(S_h[-1] - S_h[-2])) ** (1 / m)
    h_opt = (b - a) / math.ceil((b - a) / h_opt)
    h_init = h_opt
    S_h = [complexIqf(a, b, delta=h_opt), complexIqf(a, b, delta=h_opt / L)]
    h_opt /= (L ** 2)
    R_h = 1
    while abs(R_h) > eps:
        S_h.append(complexIqf(a, b, delta=h_opt))
        sol, m = err.Richardson(S_h, h_init, L=L)
        R_h = abs(sol - S_h[-1])
        print(f"h={h_opt}, погрешность: {R_h}, скорость сходимости по Эйткинсу: {m}")
        steps += 1
        h_opt /= L

