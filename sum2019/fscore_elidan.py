#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator
import openturns as ot
import hill_climbing as hc
import matplotlib.pyplot as plt

def dag_to_bn(dag, names):
    # DAG to BN
    bn = gum.BayesNet()
    for name in names:
        bn.add(gum.LabelizedVariable(name))
    for arc in dag.arcs():
        bn.addArc(arc[0], arc[1])
    return bn

max_parents = 4  # Maximum number of parents
n_restart_hc = 3 # Number of restart for the hill climbing
n_samples = 60   # Number of points calculated
n_restart = 10   # Number of restart for each point
start_size = 10
end_size = 10000

# Loading data and true structure
data_set = "data/sample_regis/sample.csv"
data_set_path = '/'.join(data_set.split('/')[:-1])
data_set_name = data_set.split('/')[-1].split('.')[0]

data = np.loadtxt(data_set, delimiter=',', skiprows=1)

Tstruct_file_path = "data/sample_regis/struct.txt"
Tstruct_file_name = Tstruct_file_path.split('/')[-1].split('.')[0]

with open(Tstruct_file_path, 'r') as file:
    arcs = file.read().replace('\n', '')
Tstruct = gum.fastBN(arcs)    


# Computing scores
sizes = np.linspace(start_size, end_size, n_samples, dtype=int)
list_structures = []
for size in sizes:
    print(size)
    list_restart = []
    for i in range(n_restart):
        sample = data[np.random.randint(0, len(data), size=size)]
        sample = ot.Sample(sample)
        c, g, s = hc.hill_climbing(sample, max_parents, n_restart_hc)
        list_restart.append(g)
    list_structures.append(list_restart)


precision = []
recall = []
fscore = []
for l in list_structures:
    list_fscore = []
    list_recall = []
    list_precision = []
    for s in l: 
        bn = dag_to_bn(s, Tstruct.names())
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


# Writting and plotting figures
header = "Size, Precision, Recall, Fscore"
title = "fscore_elidan_"  + data_set_name + "_" + "r" + str(n_restart) + \
        "mp" + str(max_parents) + "s" + str(n_samples) + "f" + str(start_size) + \
        "e" + str(end_size)

np.savetxt(data_set_path + '/' + title + ".csv",
           results, fmt="%f", delimiter=',', header=header)
 
plt.plot(sizes, precision, label='precision')
plt.plot(sizes, recall, label='recall')
plt.plot(sizes, fscore, label='fscore')
plt.legend()
plt.savefig(data_set_path + '/' + title + ".pdf", transparent=True)
plt.show()