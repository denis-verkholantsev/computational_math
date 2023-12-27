import numpy as np

def gauss_seidel(A, b, x0=0, tol=1e-8, max_iter=1000):
    if x0 == 0:
         x0 = np.zeros_like(b)
    n = len(A)
    x = x0.copy()

    # Compute the spectral radius of A
    rho = np.max(np.abs(np.linalg.eigvals(A)))

    if rho >= 1:
        raise ValueError("The method may not converge")

    for k in range(max_iter):
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x[i] = (b[i] - s) / A[i][i]

        # Check convergence based on the norm of the residual
        residual_norm = np.linalg.norm(np.dot(A, x) - b)
        if residual_norm < tol:
            return x, k

    raise ValueError("The method did not converge")