import pyAgrum as gum
import openturns as ot
import otagrum as otagr

print('Importing data')
data = ot.Sample.ImportFromTextFile('../data/Standard_coefficients_0100000.csv', ';')
data = data[0:20000]
data = data.getMarginal(range(0,12))

print('Initializing the learners')
learners = { 'cbic': otagr.TabuList(data, 3, 1, 2),
              'cpc': otagr.ContinuousPC(data, 4, 0.01),
            'cmiic': otagr.ContinuousMIIC(data) }

dags = {}
for (name,learner) in learners.items():
    print('Learning with ', name)
    dags[name] = learner.learnDAG()

for (name,dag) in dags.items():
    dot = dag.toDot()
    with open("dag_{}.dot".format(name), "w") as f:
        f.write(dot)
