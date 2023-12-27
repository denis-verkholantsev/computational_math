import numpy as np
import time
from scipy.optimize import fsolve
from solutionByLU import *
from sysAndJacobi import *

# раскладывать j один раз и не обращать матрицу в стандартном методе

#метод Ньютона
def solveNewton(error, x):
    x_0 = np.array(x, float)
    start = time.perf_counter()
    cntIter = 0
    difference = 1
    while np.any(abs(difference) > error) and cntIter < 1000:
        F = matrix(x_0) # система в точке
        J = jacobiMatrix(x_0) # матрица якоби в точке
        x_k = x_0 - np.dot(np.linalg.inv(J), F)  # x[k + 1] = x[k] - J^-1 * F
        difference = x_k - x_0
        x_0 = x_k
        cntIter += 1
    resTime = time.perf_counter() - start
    print(f"\nМетод Ньютона:\n{x_0}\n Количество итераций {cntIter}")
    print(f"{resTime} секунд")
    return x_0, resTime


# модифицированный метод Ньютона
def modifiedNewton(error, x):
    x_0 = np.array(x, float)
    start = time.perf_counter()
    cntIter = 0
    counterLU = 0
    difference = 1
    J = jacobiMatrix(x_0)
    M, q, p, counterLU = decomposeLU(J)
    L, U = getL_U(M)
    P, Q = getP_Q(p, q, M.shape[0])
    while np.any(abs(difference) > error) and cntIter < 1000:
        F = matrix(x_0)
        # dx, counter = solveLU(J, -F) # J(x[k]) * (x[k + 1] - x[k]) = -F(x[k])
        y = solveL(L, np.dot(P, -F)) #Ly = Pb
        dx = solveU(U, y) # Ux = y
        dx = np.dot(Q, dx)
        difference = dx
        x_0 += dx
        cntIter += 1
    resTime = time.perf_counter() - start
    print(f"\nМодифицированный метод Ньютона:\n{x_0}\nКоличество итераций: {cntIter}, + LU-разложения: {counterLU}")
    print(f"{resTime} секунд")
    return x_0, resTime


# Микс
def modifiedNewton_LUAfterKIterations(error, x, k):
    x_0 = np.array(x, float)
    counterLU = 0
    start = time.perf_counter()
    cntIter = 0
    difference = 1
    while np.any(abs(difference) > error) and cntIter < k:
        F = matrix(x_0)
        J = jacobiMatrix(x_0)
        x_k = x_0 - np.dot(np.linalg.inv(J), F)  # x[k + 1] = x[k] - J^-1 * F
        difference = x_k - x_0
        x_0 = x_k
        cntIter += 1
        
    J = jacobiMatrix(x_0)
    M, q, p, counterLU = decomposeLU(J)
    L, U = getL_U(M)
    P, Q = getP_Q(p, q, M.shape[0])
    while np.any(abs(difference) > error) and cntIter < 1000:
        F = matrix(x_0)
        # dx, counter = solveLU(J, -F)  # J(x[k]) * (x[k + 1] - x[k]) = -F(x[k])
        y = solveL(L, np.dot(P, -F)) #Ly = Pb
        dx = solveU(U, y) # Ux = y
        dx = np.dot(Q, dx)
        difference = dx
        x_0 += dx
        cntIter += 1
    resTime = time.perf_counter() - start
    print(f"\n Модифицированный после {k} итераций:\n{x_0}\n Количество итераций: {cntIter} + LU-разложения: {counterLU}")
    print(f"{resTime} секунд")
    return x_0, resTime


def cycleNewton(error, x, m):
    x_0 = np.array(x, float)
    start = time.perf_counter()
    cntIter = 0
    difference = 1
    while np.any(abs(difference) > error) and cntIter < 1000:
        J = jacobiMatrix(x_0)
        i = 0
        while i < m and np.any(abs(difference) > error) and cntIter < 1000:
            F = matrix(x_0)
            x_k = x_0 - np.dot(np.linalg.inv(J), F)  # x[k + 1] = x[k] - J^-1 * F
            difference = x_k - x_0
            x_0 = x_k
            cntIter += 1
            i += 1
    resTime = time.perf_counter() - start
    print(f"\nЦиклический метод при m = {m}:\n{x_0}\nКоличество итераций: {cntIter}")
    print(f"{resTime} секунд")
    return x_0, resTime


def combinedNewton(error, x, k, m):
    x_0 = np.array(x, float)
    start = time.perf_counter()
    cntIter = 0
    difference = 1
    counterLU = 0
    while np.any(abs(difference) > error) and cntIter < k:
        J = jacobiMatrix(x_0)
        i = 0
        while i < m and np.any(abs(difference) > error) and cntIter < k:
            F = matrix(x_0)
            x_k = x_0 - np.dot(np.linalg.inv(J), F)  # x[k + 1] = x[k] - J^-1 * F
            difference = x_k - x_0
            x_0 = x_k
            cntIter += 1
            i += 1

    while np.any(abs(difference) > error) and cntIter < 1000:
        # cчитаем и раскладываем матрицу якоби
        J = jacobiMatrix(x_0)
        M, q, p, counter = decomposeLU(J)
        L, U = getL_U(M)
        P, Q = getP_Q(p, q, M.shape[0])
        counterLU += counter
        i = 0
        while i < m and np.any(abs(difference) > error) and cntIter < 1000:
            # решаем систему с уже разложенной, то есть исключаем дополнительные m-1 разложение за O(n^3)
            F = matrix(x_0)
            # dx, counter = solveLU(J, -F)  # J(x[k]) * (x[k + 1] - x[k]) = -F(x[k])
            y = solveL(L, np.dot(P, -F)) #Ly = Pb
            dx = solveU(U, y) # Ux = y
            dx = np.dot(Q, dx)
            difference = dx
            x_0 += dx
            cntIter += 1
            i += 1
    
    # J = jacobiMatrix(x_0)
    # M, q, p, counter = decomposeLU(J)
    # L, U = getL_U(M)
    # P, Q = getP_Q(p, q, M.shape[0])
    # counterLU += counter
    # while np.any(abs(difference) > error) and cntIter < 1000:
    #     # решаем систему с уже разложенной, то есть исключаем дополнительные m-1 разложение за O(n^3)
    #     F = matrix(x_0)
    #     # dx, counter = solveLU(J, -F)  # J(x[k]) * (x[k + 1] - x[k]) = -F(x[k])
    #     y = solveL(L, np.dot(P, -F)) #Ly = Pb
    #     dx = solveU(U, y) # Ux = y
    #     dx = np.dot(Q, dx)
    #     difference = dx
    #     x_0 += dx
    #     cntIter += 1
    
    resTime = time.perf_counter() - start
    print(f"\nКомбинированный метод\n{x_0}\nКоличество итераций: {cntIter} + LU-разложения: {counterLU}")
    print(f"{resTime} секунд")
    return x_0, resTime


def scpyNewton(x_0):
    start = time.perf_counter()
    x_0 = fsolve(matrixForList, x_0)
    resTime = time.perf_counter() - start
    print(f"Scpy:\n{np.array(x_0).reshape(10, 1)}")
    print(f"{resTime} секунд")
    return x_0, resTime