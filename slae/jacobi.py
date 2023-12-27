import numpy as np

def jacobi(A, b, x0=0, tol=1e-8, max_iter=1000):
    if x0 == 0:
         x0 = np.zeros_like(b)
    n = len(A)
    x = x0.copy()

    # Compute the spectral radius of D^-1 * (L+U)
    D_inv = np.diag(1 / np.diag(A))
    L_plus_U = A - np.diag(np.diag(A))
    rho = np.abs(np.linalg.norm(-np.dot(D_inv, L_plus_U)))

    if rho >= 1:
        raise ValueError("Метод не сойдется")

    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = np.dot(A[i], x) - A[i][i] * x[i]
            x_new[i] = (b[i] - s) / A[i][i]
        
        if np.linalg.norm(x_new - x) <= (1-rho)/rho*tol:
            return x_new, k
        x = x_new
        
    raise ValueError("Метод не сошелся")

def pseudo_jacobi(M, b, x, eps=1e-10, max_iter=1000):
    D = np.diag(M)
    R = M - np.diagflat(D)

    for i in range(max_iter):
        prevX = x.copy()
        x = (b - np.dot(R, prevX)) / D
        if np.linalg.norm(x - prevX) < eps:
            print("Метод сошелся за" + str(i) + " итераций.")
            return x

    return "Метод Якоби не сошелся."