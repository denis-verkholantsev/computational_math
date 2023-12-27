from LU import inverse_matrix, cond_number, solveLU, get_permutation_matrixes_through_vectors, decomposeLU, rank_matrix, determinantLU, getLU
import numpy as np


def test_random(numOfMatrixes):
    for i in range(numOfMatrixes):
        random_size = np.random.randint(3, 10)
        A = np.random.uniform(low=-10.5, high=23.1, size = (random_size, random_size))
        b = np.random.uniform(low=-10.5,high=23.1, size = random_size)
        print(f'{i+1} MATRIX',A, "b:", b, sep='\n')
        print("SLAE:", solveLU(A,b),sep='\n')
        print("rank:", rank_matrix(A))
        print("inversed:", inverse_matrix(A),sep='\n')
        print("det:", determinantLU(A))
        print("conditional number:", cond_number(A))


def test_LU_equal_PAQ():
    n = np.random.randint(2, 10)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, n))
    M, colsPermutation, rowsPermutation = decomposeLU(A)
    L, U = getLU(M)
    rowsPermutationMatrix, colsPermutationMatrix = get_permutation_matrixes_through_vectors(rowsPermutation, colsPermutation, M.shape[0])
    error = 1e-9
    result = np.linalg.norm(np.dot(np.dot(rowsPermutationMatrix, A), colsPermutationMatrix) - np.dot(L, U))
    print("LU~PAQ" if result < error else "LU!~PAQ")


def test_determinant():
    n = np.random.randint(2, 10)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, n))
    error = 1e-9
    print("det(LU)=det(A)" if abs(determinantLU(A) - np.linalg.det(A)) < error else "det(LU)!=det(A)")


def test_inversed():
    n = np.random.randint(2, 3)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, n))
    invA = inverse_matrix(A)
    E = np.eye(n)
    error = 1e-9
    print(np.dot(A, invA))
    result1 = np.linalg.norm(np.dot(A, invA) - E)
    result2 = np.linalg.norm(np.dot(invA, A) - E)
    print(result1, result2)
    print("Обратная верна" if result1 < error and result2 < error else "Обратная неверна")


def test_rank():
    n = np.random.randint(2, 10)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, n))
    print("Ранг найден верно" if rank_matrix(A) == np.linalg.matrix_rank(A) else "Ранг найден неверно")
