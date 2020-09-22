import openturns as ot
import otagrum as otagr
import numpy as np
import time
import matplotlib.pyplot as plt

correlation = 0.8

sizes = [10, 100, 1000, 10000, 100000]
dimensions = np.arange(2, 10)

T = []
for d in dimensions:
    print("Dimension: ", d)
    cm = np.reshape([correlation] * d**2, (d,d))
    np.fill_diagonal(cm, 1)
    cm = ot.CorrelationMatrix(cm)
    normal_copula = ot.NormalCopula(cm)
    normal_sample = normal_copula.getSample(1000000)
    t = []
    for s in sizes:
        print("    Size: ", s)
        sample = normal_sample[:s]
        K = otagr.ContinuousTTest_GetK(s, 2)

        start = time.time()
        bc = ot.EmpiricalBernsteinCopula(sample, K, False)
        end = time.time()

        t.append(end-start)
    T.append(t)
T = np.array(T)

for i in range(len(dimensions)):
    plt.plot(sizes, T[i], label=str(dimensions[i])+'D')
plt.legend()
plt.show()

for i in range(len(sizes)):
    plt.plot(dimensions, T.T[i], label=(str(sizes[i])))
plt.legend()
plt.show()
