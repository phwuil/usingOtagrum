import numpy as np
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

# Sampling from standard normal
D = ot.Normal([0] * N, [1] * N, R).getSample(M)

# Switch to rank space
D_r = (D.rank()+1)/(D.getSize()+2)

G, S = hc.hill_climbing(D_r,2)

print("G: ", G)
print("S: ", S)