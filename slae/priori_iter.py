import numpy as np
import math

def priori_iter(A, b, x0=0, epsilon = 1e-8):
    if x0==0:
        x0 = np.zeros_like(b)
    n = len(A)
    D_inv = np.diag(1 / np.diag(A))
    L_plus_U = A - np.diag(np.diag(A))
    rho = np.max(np.abs(np.linalg.eigvals(np.dot(D_inv, L_plus_U))))
    if rho < 1:
        k = math.log(epsilon*(1-rho)/np.linalg.norm(x0),rho)
        return k
    else:
        print("Метод не сойдется")