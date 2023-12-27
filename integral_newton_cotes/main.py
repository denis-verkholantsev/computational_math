import numpy as np
from func import *
import NewtonCotes
import Gauss


def Gauss_(a_, b_, eps):
    a = 3.2 - b_
    b = 3.2 - a_
    print("Методы Гаусса: ")
    print("Результат ИКФ:")
    print(Gauss.iqf(a, b))
    print("\n")
    print("Результат CИКФ: ", end='\n')
    print(Gauss.complexIqf(a, b, 100))
    print("\n")
    print("Результат CИКФ c точностью " + str(eps), end='\n')
    Gauss.complexIqf_err(a, b, eps)
    print("\n")
    print("Результат CИКФ c выбором оптимального шага с точностью " + str(eps), end='\n')
    Gauss.complexIqf_opt(a, b, eps)

def Newton_Cotes_(a_, b_, eps):
    a = 3.2 - b_
    b = 3.2 - a_
    print("Методы Ньютона-Котса: ")
    print("Результат ИКФ: ", end='\n')
    print(NewtonCotes.iqf(a, b))
    print("\n")
    print("Результат CИКФ: ", end='\n')
    print(NewtonCotes.complexIqf(a, b, steps = 7))
    print("\n")
    print("Результат CИКФ c точностью " + str(eps), end='\n')
    NewtonCotes.complexIqf_err(a, b, eps)
    print('\n')
    print("Результат CИКФ c выбором оптимального шага с точностью " + str(eps), end='\n')
    NewtonCotes.complexIqf_opt(a, b, eps)
    print('\n')


val = 11.83933565874812191864851199716726555747


if __name__ == '__main__':
    a = 1.7
    b = 3.2
    eps = 1e-9
    print(val)
    Newton_Cotes_(a,b, eps)
    # Gauss_(a, b, eps)