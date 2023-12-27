import numpy as np
from QR import givensQR, decompositionQR

def testDecompQR():
    n = np.random.randint(2,10)
    m = np.random.randint(2,10)
    A = np.random.uniform(low=-10.5, high=23.1, size=(n, m))
    Q1, R1 = givensQR(A)
    Q2, R2 = decompositionQR(A)
    error = 1e-9
    print("Гивенс - верно" if np.linalg.norm(np.dot(Q1, R1) - A, ord = 1) < error else "Гивенс - неверно")
    print("ГР-ШМ - верно" if np.linalg.norm(np.dot(Q2, R2) - A, ord = 1) < error else "ГР-ШМ - неверно")