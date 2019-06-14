# doc
# http://openturns.github.io/openturns/master/index.html
# 
import numpy as np
#import pyAgrum as gum
import openturns as ot
import hill_climbing as hc
#import otagrum as otagr

ot.RandomGenerator.SetSeed(42)
np.random.seed(42)

M = 100 # Size of the dataset
N = 4  # Dimension of the random vector

# Correlation matrix definition
R = ot.CorrelationMatrix(N)
R[0,1] = 0.3
R[2,3] = -0.3

print(R)

C = ot.NormalCopula(R)
I = ot.IndependentCopula(N)

# Sampling from standard normal
D = ot.Normal([0] * N, [1] * N, R).getSample(M)
D_r = (D.rank()+1)/(D.getSize()+2)

#D_r = C.getSample(M)

print("D : ", D)
print("D_r : ", D_r)

G, S = hc.hill_climbing(D_r)

print(G, S)