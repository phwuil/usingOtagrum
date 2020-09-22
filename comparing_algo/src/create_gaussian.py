import openturns as ot
import otagrum
import pyAgrum as gum
import numpy as np

def buildT(b):
    dim = len(b)
    T = ot.CovarianceMatrix(dim)
    T[0, 0] = 1.0
    for i in range(1, dim):
        for j in range(i):
            for k in range(j+1):
                T[j, k] += b[i][j] * b[i][k]
            T[i, j] = -b[i][j]
        T[i, i] = 1.0
    return T

proto = "X1->X2<-X3->X5->X6;X2->X4<-X5"
bn = gum.BayesNet.fastPrototype(proto)
print("      proto : ", proto)
print("         BN : ", bn)

ndag = otagrum.NamedDAG(bn)

f = open("bn.dot", "w")
f.write(ndag.toDot())
f.close()

order = ndag.getTopologicalOrder()
print("order=", order)

description = ndag.getDescription()
print("description=", description)

for node in order:
    print(" parents(", description[node], ") : ",
          [description[i] for i in ndag.getParents(node)])
    print("children(", description[node], ") : ",
          [description[i] for i in ndag.getChildren(node)])

# Conditional linear coefficients, b matrix:
# X3, X5, X6, X1, X2, X4
b = [[],
     [0.5],
     [0.0, 0.75],
     [0.0, 0.00, 0.0],
     [0.5, 0.00, 0.0, 0.75],
     [0.0, 1.00, 0.0, 0.00, 1.0]]

T = buildT(b)
I = ot.IdentityMatrix(T.getDimension())
sigma = ot.CovarianceMatrix(np.array(T.solveLinearSystem(I)))

sigma_ordered = ot.CovarianceMatrix(len(b))
for i in range(len(b)):
    for j in range(i+1):
        sigma_ordered[order[i], order[j]] = sigma[i, j]
print("Sigma matrix: ", sigma_ordered)

# Marginal mean, mu vector:
mu = [0.0]*6

distribution = ot.Normal(mu, sigma_ordered)
distribution.setDescription(description)
size = 100000
sample = distribution.getSample(size)
sample.exportToCSVFile("../data/sample.csv")

# print("ContinuousPC")
# alpha = 0.1
# binNumber = 4
# learner = otagrum.ContinuousPC(sample, binNumber, alpha)
# learner.setVerbosity(True)
# pdag = learner.learnPDAG()
# print(learner.PDAGtoDot(pdag))
# dag = learner.learnDAG()
# print(dag.toDot())

print("ContinuousMIIC")
learner = otagrum.ContinuousMIIC(sample)
learner.setCMode(otagrum.CorrectedMutualInformation.CModeTypes_Gaussian)
learner.setVerbosity(True)
dag = learner.learnDAG()
print(dag.toDot())
