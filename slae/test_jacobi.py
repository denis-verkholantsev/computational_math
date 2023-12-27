from jacobi import jacobi

def testJacobiDiagNotPos():
    n = np.random.randint(5, 10)
    A = []
    A = np.random.uniform(low=-10, high=23.1, size=(n, n))
    for i in range(n):
        sumWithoutI = 0
        for j in range(n):
            if i!=j:
                sumWithoutI+=A[i][j]
        if sumWithoutI > A[i][i]:
            A[i][i] = sumWithoutI * 2
    b = np.random.uniform(low=-10, high=23.1, size=(n, 1))
    if np.linalg.det(A) != 0:
        x, k = jacobi(A, b)
        print("solution: ", x)
        print("k:", k)

testJacobiDiagNotPos()

import numpy as np

def testJacobiPos():
    n = np.random.randint(5, 10)
    A = []
    flag = True
    while flag:
        A = np.random.uniform(low=-10, high=23.1, size=(n, n))
        for i in range(n):
            sumWithoutI = 0
            for j in range(n):
                if i!=j:
                    sumWithoutI += A[i][j]
            if sumWithoutI < A[i][i]:
                A[i][i] = sumWithoutI / 2
        eigs = np.linalg.eigvals(A)
        for eig in eigs:
            if eig < 0:
                flag = False
    b = np.random.uniform(low=-10, high=23.1, size=(n, 1))
    if np.linalg.det(A) != 0:
        x, k = jacobi(A, b)
        print("solution: ", x)
        print("k:", k)