import numpy as np
from math import *


def f(x):
    x = 3.2 - x
    return 3 * np.cos(0.5 * x) * np.exp(x / 4) + 5 * np.sin(2.5 * x) * np.exp(-x / 3) + 2 * x

def mu_i(a, b, i):
    return (b ** (i + 0.75) - a ** (i + 0.75)) / (i + 0.75)



