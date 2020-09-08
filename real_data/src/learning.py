import pyAgrum as gum
import openturns as ot
import otagrum as otagr

data = ot.Sample.ImportFromTextFile('../data/Standard_coefficients_0100000.csv', ';')
data = data[0:10000]
data = data.getMarginal(range(0,12))

learner = otagr.TabuList(data, 2, 1, 2)
learner.setVerbosity(True)
dag = learner.learnDAG()
string = dag.toDot()

with open("output_cpc.txt", "w") as fo:
    fo.write(string)
