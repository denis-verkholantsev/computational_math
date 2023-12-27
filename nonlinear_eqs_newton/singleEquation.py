import numpy as np


def derivative(f, x, h):
    return (f(x+h) - f(x)) / (h)


def newton(f, a, b, eps, maxIter = 100):
    if f(a)*f(b) > 0:
        print("На данном отрезке корень отсутствует")
        return
    x = (a + b) / 2
    for _ in range(maxIter):
        dfdx = derivative(f, x, eps)
        if abs(f(x) /dfdx) < eps:
            break
        x = x - f(x) / dfdx
        if x < a or x > b: # если вышел за границы
            return bisect(f, a, b, eps, maxIter)
    if abs(f(x)/dfdx) < eps:
        return x
    else:
        print("Метод Ньютона не сошелся или максимум итераций")        


def bisect(f, a, b, eps, maxIter = 100):
    if f(a)*f(b) > 0:
        print("На данном отрезке корень отсутствует")
        return
    c = 0
    for _ in range(maxIter):
        c = (a + b) / 2
        if abs(f(c)) < eps:
            break
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    if abs(f(c)) < eps:
        return c
    else:
        print("Метод бисекций не сошелся или максимум итераций")

