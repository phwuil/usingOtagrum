import openturns as ot
import otagrum as otagr

X = ot.Sample.ImportFromCSVFile("advised.csv")
binNumber = 2
alpha = 0.9
learner = otagr.ContinuousPC(X, binNumber, alpha)
res = learner.learnDAG()
print(res.toDot())
