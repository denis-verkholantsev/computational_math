import numpy as np

def getL_U(A):
    L = np.tril(A, -1)
    for i in range(L.shape[0]):
        L[i, i] = 1
    U = np.triu(A)
    return L, U


def getP_Q(p, q, sizeA):
    P = np.eye(sizeA) #матрица перестановок строк P
    for i in range(p.shape[0]):
        P[[p[i], i], :] = P[[i, p[i]], :]

    Q = np.eye(sizeA) #матрица перестановок столбцов
    for i in range(q.shape[0]):
       Q[:, [q[i], i]] =Q[:, [i, q[i]]]

    return P, Q


def gauss(M):
    t = np.array(M, float)
    t = t[1:] / t[0]
    return t


def gauss_app(M, t):
    C = np.array(M, float)
    t = np.array([[t[i]] for i in range(len(t))])
    C[1:, :] = C[1:, :] - t * C[0, :]
    return C


def decomposeLU(M):
    A = np.array(M, float)
    size = A.shape[0]
    q = np.arange(0, size)
    p = np.arange(0, size)
    counter = 0
    for k in range(size - 1):
        indicesMax = np.unravel_index(np.argmax(abs(A[k:, k:]), axis=None), A[k:, k:].shape)  #  координаты наибольшего элемента
        q[k] = indicesMax[1] + k # векторы перестановок
        p[k] = indicesMax[0] + k
        A[[k, p[k]], :] = A[[p[k], k], :] #свап
        A[:, [k, q[k]]] = A[:, [q[k], k]]
        if A[k, k] != 0:
            t = gauss(A[k:, k]) # вектор множителей
            A[k:, k + 1:] = gauss_app(A[k:, k + 1:], t) # преобразование Гаусса
            A[k + 1:, k] = t # элементы равны множителям Гаусса
            counter += 1
        else:
            return A, q, p, counter
    return A, q, p, counter


def solveL(L, b):
    b[0] /= L[0, 0]
    for i in range(1, len(b)):
        b[i] = (b[i] - np.dot(L[i, :i], b[:i])) / L[i, i]
    return b


def solveU(U, b):
    if U[-1, -1]:
        b[-1] /= U[-1, -1]
    else:
        b[-1] = 0
    for k in range(len(b) - 2, -1, -1):
        if U[k, k]:
            b[k] = (b[k] - np.dot(U[k, k + 1:], b[k + 1:])) / U[k, k]
        else:
            b[k] = 0
    return b


def inverseMatrix(A):
    M, q, p = decomposeLU(A)
    L, U = getL_U(M)
    P, Q = getP_Q(p, q, M.shape[0])
    result = np.eye(M.shape[0])
    for k in range(M.shape[0]):
        result[:, k] = solveL(L, result[:, k])

    for k in range(M.shape[0]):
        result[:, k] = solveU(U, result[:, k])

    return np.dot(np.dot(Q, result), P)


def solveLU(A, b):
    M, q, p,counter = decomposeLU(A)
    L, U = getL_U(M)
    P, Q = getP_Q(p, q, M.shape[0])
    y = solveL(L, np.dot(P, b)) #Ly = Pb
    x = solveU(U, y) # Ux = y
    return np.dot(Q, x), counter