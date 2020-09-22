import data_generation as dg
import otagrum
import pyAgrum as gum
import matplotlib.pyplot as plt

ds_size = 10000
distribution = 'student'
restarts = 20

S = list(range(1000, 10100, 100))

names = ['X', 'Y']

dag = gum.DAG()
dag.addNodes(2)
# dag.addArc(0,1)

ndag = otagrum.NamedDAG(dag, names)

D = [dg.generate_data(ndag, ds_size, distribution, r=0.8) for _ in range(restarts)]

I = []
for size in S:
    print("Size: ", size)
    info = 0
    for i,data in enumerate(D):
        print("Restart: ", i+1)
        cmi = otagrum.CorrectedMutualInformation(data[:size])
        cmi.setKMode(otagrum.CorrectedMutualInformation.KModeTypes_NoCorr)
        info += cmi.compute2PtCorrectedInformation(0, 1)
    I.append(info/restarts)

plt.plot(S, I)
plt.show()
