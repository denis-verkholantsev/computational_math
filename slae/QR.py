import numpy as np

def decompositionQR(A):
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def givensQR(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        for i in range(m-1, j, -1):
            a = R[i-1, j]
            b = R[i, j]
            r = np.sqrt(a**2 + b**2)
            c = a / r
            s = -b / r
            G = np.eye(m)
            G[[i-1, i], [i-1, i]] = c
            G[i-1, i] = s
            G[i, i-1] = -s
            R = np.dot(G, R)
            Q = np.dot(Q, G.T)
    return Q, R


def solve_QR(A, b):
    Q, R = givensQR(A)
    y = np.dot(Q.T, b)
    n = R.shape[0]
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(R[i,i+1:], x[i+1:]))/R[i,i]
    return x
