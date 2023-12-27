import system
from singleEquation import *
from sysAndJacobi import *


def testSingleEq():
   fun = lambda x: x ** 3 - np.exp(x) + 1
   pairs = [[1,2],[-1,-0.5],[-0.5,1],[3,8]]
   res = []
   for ab in pairs:
      res.append(newton(fun, a=ab[0], b=ab[1], eps=1e-10, maxIter=1000))   
   print("Решение уравнения: ", res)
   print("\n---------------------------------")


def testFirstInitial():
   x_0 = np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5], float).reshape(10, 1)
   error = 1e-6
   x0, time0 = system.scpyNewton(x_0.reshape(x_0.shape[1], x_0.shape[0]))
   x1, time1 = system.solveNewton(error, x_0)
   x2, time2 = system.modifiedNewton(error, x_0)
   x3, time3 = system.modifiedNewton_LUAfterKIterations(error, x_0, k=4)
   x4, time4 = system.cycleNewton(error, x_0, m=5)
   x5, time5 = system.combinedNewton(error, x_0, k=8, m=5)


def testSecondInitial():
   print("\n\n---------Второе начальное приближение-------------\n\n")
   error = 1e-6
   x_0 = np.array([0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5], float).reshape(10, 1)
   x0_, time0 = system.scpyNewton(x_0.reshape(x_0.shape[1], x_0.shape[0]))
   x1_, time1_ = system.solveNewton(error, x_0)
   x2_, time2_ = system.modifiedNewton(error, x_0)
   x3_, time3_ = system.modifiedNewton_LUAfterKIterations(error, x_0, k=7) # k=6 k>7 разные
   x4_, time4_ = system.cycleNewton(error, x_0, m = 7) 
   x5_, time5_ = system.combinedNewton(error, x_0, k=5, m=2) # аналогично


def testingMod2CycleCombined():
   error = 1e-6
   x_0 = np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5], float).reshape(10, 1)
   x1, _ = system.solveNewton(error, x_0)
   bestTime3 = 100
   bestK3 = 0
   for i in range(20):
      x, time = system.modifiedNewton_LUAfterKIterations(error, x_0, i)
      if np.any(abs(x - x1) < error):
         bestTime3 = min(bestTime3, time)
         bestK = i
      print(bestTime3, bestK)
   print("------------------------------")
   bestTime4 = 100
   bestM4 = 0
   for i in range(20):
      x, time = system.cycleNewton(error, x_0, i+1)
      if np.any(abs(x - x1) < error):
         bestTime4 = min(bestTime4, time)
         bestM4 = i+1
      print(bestTime4, bestM4)
   print("------------------------")
   bestTime5 = 100
   bestK5 = 0
   bestM5 = 0
   for i in range(20):
      for j in range(20):
         x, time = system.combinedNewton(error, x_0, i+1, j+1)
         if np.any(abs(x - x1) < error):
               bestTime5 = min(bestTime5, time)
               bestK5 = i+1
               bestM5 = j+1
               print(bestTime5, bestK5,bestM5)  
   print("3: ", bestTime3, bestK3)
   print("4: ", bestTime4, bestM4)
   print("5: ", bestTime5, bestK5, bestM5)
