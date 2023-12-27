import numpy as np
import math


def Aitken(L, S_h):
    return -math.log(math.fabs((S_h[-1] - S_h[-2]) / (S_h[-2] - S_h[-3]))) / math.log(L)


def Richardson(S_h, h, L):
    m = Aitken(L, S_h[-3:])
    size = len(S_h)
    A = np.zeros((size,size))
    for i in range(size):
        A[i][-1] = -1  # последний столбец J(f)
        for j in range(size - 1):
            A[i][j] = (h / (L ** i)) ** (m + j)
    solution = np.linalg.solve(A, -np.array(S_h))
    return abs(solution[-1]), m

