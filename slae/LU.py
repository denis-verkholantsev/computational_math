import numpy as np

def getLU(A):
    L = np.tril(A, -1)
    for i in range(L.shape[0]):
        L[i, i] = 1
    U = np.triu(A)
    return L, U


def get_permutation_matrixes_through_vectors(rowsPermutation, colsPermutation, sizeA):
    rowsPermutationMatrix = np.eye(sizeA) #матрица перестановок строк P
    for i in range(rowsPermutation.shape[0]):
        rowsPermutationMatrix[[rowsPermutation[i], i], :] = rowsPermutationMatrix[[i, rowsPermutation[i]], :]

    colsPermutationMatrix = np.eye(sizeA) #матрица перестановок столбцов
    for i in range(colsPermutation.shape[0]):
       colsPermutationMatrix[:, [colsPermutation[i], i]] = colsPermutationMatrix[:, [i, colsPermutation[i]]]

    return rowsPermutationMatrix, colsPermutationMatrix


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
    cols_permutation = np.arange(0, size)
    rows_permutation = np.arange(0, size)
    k = 0
    while k < size - 1:
        indicesMax = np.unravel_index(np.argmax(abs(A[k:, k:]), axis=None), A[k:, k:].shape)  #  координаты наибольшего элемента
        cols_permutation[k] = indicesMax[1] + k # векторы перестановок
        rows_permutation[k] = indicesMax[0] + k
        A[[k, rows_permutation[k]], :] = A[[rows_permutation[k], k], :] #свап
        A[:, [k, cols_permutation[k]]] = A[:, [cols_permutation[k], k]]
        if A[k, k] != 0:
            t = gauss(A[k:, k]) #вектор множителей
            A[k:, k + 1:] = gauss_app(A[k:, k + 1:], t) #преобразование Гаусса
            A[k + 1:, k] = t # элементы равны множителям Гаусса
        else:
            return A, cols_permutation, rows_permutation
        k += 1
    return A, cols_permutation, rows_permutation


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


def determinantLU(A):
    M, colsPermutation, rowsPermutation = decomposeLU(A)
    sizeM = M.shape[0]
    result = M.diagonal().prod()
    rowsPermutationMatrix, colsPermutationMatrix = get_permutation_matrixes_through_vectors(rowsPermutation, colsPermutation, sizeM)
    result *= np.linalg.det(rowsPermutationMatrix) * np.linalg.det(colsPermutationMatrix)
    return result


def inverse_matrix(A):
    M, colsPermutation, rowsPermutation = decomposeLU(A)
    L, U = getLU(M)
    rowsPermutationMatrix, colsPermutationMatrix = get_permutation_matrixes_through_vectors(rowsPermutation, colsPermutation, M.shape[0])
    result = np.eye(M.shape[0])
    for k in range(M.shape[0]):
        result[:, k] = solveL(L, result[:, k])

    for k in range(M.shape[0]):
        result[:, k] = solveU(U, result[:, k])

    return np.dot(np.dot(colsPermutationMatrix, result), rowsPermutationMatrix)


def cond_number(A):
    M = np.array(A, float)
    norm = np.linalg.norm(M, ord=1)
    inv_norm = np.linalg.norm(inverse_matrix(M), ord=1)
    result = norm * inv_norm
    if result >= 1:
        return result
    return "Неверное число обусловленности"


def rank_matrix(M):
    A = np.array(M,float)
    error = 1e-10
    LU, p, q = decomposeLU(A)
    U = np.triu(LU)
    result = 0
    for i in range(U.shape[0]):
        if abs(U[i, i]) < error:
            result = i
            break
        else:
            result += 1
    return result


def solveLU(A, b):
    M, colsPermutation, rowsPermutation = decomposeLU(A)
    L, U = getLU(M)
    rowsPermutationMatrix, colsPermutationMatrix = get_permutation_matrixes_through_vectors(rowsPermutation, colsPermutation, M.shape[0])
    y = solveL(L, np.dot(rowsPermutationMatrix, b)) #Ly = Pb
    rowInd = y.shape[0] - 1

    while y[rowInd] == 0:
        if U[rowInd, U.shape[1] - 1] != 0:
            return "Нет решений."
        rowInd -= 1

    x = solveU(U, y) # Ux = y
    return np.dot(colsPermutationMatrix, x)