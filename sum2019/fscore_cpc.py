#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator
import openturns as ot
import otagrum as otagr
import matplotlib.pyplot as plt

def named_dag_to_bn(ndag, names):
    # DAG to BN
    bn = gum.BayesNet()
    names = ndag.getDescription()
    for name in names:
        bn.add(gum.LabelizedVariable(name))
    for node in ndag.getNodes():
        for child in ndag.getChildren(node):
            bn.addArc(names[node], names[child])
    return bn

binNumber = 4
alpha = 0.1
n_samples = 50
n_restart = 10
start_size = 10
end_size = 50000

# Loading of data and true structure
data_set = "data/sample_regis/sample.csv"
data_set_path = '/'.join(data_set.split('/')[:-1])
data_set_name = data_set.split('/')[-1].split('.')[0]

data = np.loadtxt(data_set, delimiter=',', skiprows=1)

Tstruct_file_path = "data/sample_regis/struct.txt"
Tstruct_file_name = Tstruct_file_path.split('/')[-1].split('.')[0]

with open(Tstruct_file_path, 'r') as file:
    arcs = file.read().replace('\n', '')
Tstruct = gum.fastBN(arcs)

sizes = np.linspace(start_size, end_size, n_samples, dtype=int)
list_structures = []
for size in sizes:
    print(size)
    list_restart = []
    for i in range(n_restart):
        sample = data[np.random.randint(0, len(data), size=size)]
        sample = ot.Sample(sample)
        sample.setDescription(Tstruct.names())
        learner = otagr.ContinuousPC(sample, binNumber, alpha)
        t = otagr.ContinuousTTest(sample)
        #print("B indep E | A", t.isIndep(1,4,[0]))
        #print("A indep D | C", t.isIndep(0,3,[2]))
        #print("E indep A", t.isIndep(4,0,[]))
        list_restart.append(learner.learnDAG())
    list_structures.append(list_restart)


precision = []
recall = []
fscore = []
for l in list_structures:
    list_fscore = []
    list_recall = []
    list_precision = []
    for s in l: 
        bn = named_dag_to_bn(s, Tstruct.names())
        comparison = GraphicalBNComparator(Tstruct, bn)
        scores = comparison.scores()
        list_precision.append(scores['precision'])
        list_recall.append(scores['recall'])
        list_fscore.append(scores['fscore'])
    precision.append(np.mean(list_precision))
    recall.append(np.mean(list_recall))
    fscore.append(np.mean(list_fscore))
 
precision = np.reshape(precision, (n_samples, 1))
recall = np.reshape(recall, (n_samples,1))
fscore = np.reshape(fscore, (n_samples,1))
sizes = sizes.reshape(n_samples,1)
results = np.concatenate((sizes, precision, recall, fscore), axis=1)

header = "Size, Precision, Recall, Fscore"
title = "fscore_cpc_"  + data_set_name + "_" + "r" + str(n_restart) + \
        "spms" + str(binNumber) + "s" + str(n_samples) + "f" + str(start_size) + \
        "e" + str(end_size)

np.savetxt(data_set_path + '/' + title + ".csv",
           results, fmt="%f", delimiter=',', header=header)
    
plt.plot(sizes, precision, label='precision')
plt.plot(sizes, recall, label='recall')
plt.plot(sizes, fscore, label='fscore')
plt.legend()
plt.savefig(data_set_path + '/' + title + ".pdf", transparent=True)
plt.show()
