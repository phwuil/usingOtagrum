# -*- coding: utf-8 -*-

# Evolution of mutual information with respect to the sample size.

import numpy as np
import openturns as ot
import cmiic.continuousMIIC as cmiic

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

sizes = np.linspace(1000, 20000, 20, dtype=int)

data_vs = ot.Sample.ImportFromTextFile("../data/samples/dirichlet/vStruct/sample01.csv", ',')[:20000]
# data_vs = (data_vs.rank()+1)/(data_vs.getSize()+2)

list_01 = []
list_02 = []
list_12 = []
list_01_2 = []
for size in sizes:
    data = data_vs[0:size]
    print('Size : ', size)
    learner_vs = cmiic.ContinuousMIIC(data)
    pdag_vs = learner_vs.learnMixedStructure()
    dag_vs = learner_vs.learnStructure()
    cache = learner_vs.getIcache()
    
    I_01 = cache[(frozenset({0,1}), frozenset({}))]
    I_02 = cache[(frozenset({0,2}), frozenset({}))]
    I_12 = cache[(frozenset({1,2}), frozenset({}))]
    I_01_2 = learner_vs._ContinuousMIIC__compute2PtInformation(0,1,[2])
    
    list_01.append(I_01)
    list_02.append(I_02)
    list_12.append(I_12)
    list_01_2.append(I_01_2)
    
    
fig, ax = plt.subplots()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xlim([1000, 20000])
ax.plot(sizes, list_01, label='IAB')
ax.plot(sizes, list_02, label='IAC')
ax.plot(sizes, list_12, label='IBC')
ax.plot(sizes, list_01_2, label='IAB|C')
ax.legend()
fig.savefig("bad_info.pdf", transparent=True)