#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator
import openturns as ot
import otagrum as otagr
import matplotlib.pyplot as plt
import itertools as it

def plot_pvalues(sizes, pvalues, save=False, scale=0.1):
    x = [e for e in sizes for i in range(len(pvalues[0]))] 
    y = pvalues.flatten()
    plt.scatter(x, y, s=scale)
    if save:
        title = "pvalues_cpc_"  + data_file_name + "_" + "r" + str(n_restart) + \
        "spms" + str(binNumber) + "alpha" + str(int(100*alpha)) + \
        "s" + str(n_samples) + "f" + str(start_size) + "e" + str(end_size)
        plt.savefig(data_directory + title + ".pdf", transparent=True)
    plt.show()

def plot_error(x, mean, std, alpha=0.4):
    x, mean, std = x.flatten(), mean.flatten(), std.flatten()
    lower, upper = mean-std, mean+std
    plt.fill_between(x, lower, upper, alpha=alpha)

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

def load_struct(file):
    with open(file, 'r') as f:
        arcs = f.read().replace('\n', '')
    return gum.fastBN(arcs)

binNumber = 5
alpha = 0.05
n_samples = 30
n_restart = 20
start_size = 10
end_size = 100000

# Loading of data and true structure
data_directory = "data/gaussian/alarm/r08/"
struct_directory = "data/gaussian/alarm/"

data_file = "alarm_gaussian_sample_01.csv"
data_file_name = data_file.split('.')[0]

Tstruct_file = "alarm.txt"
Tstruct_file_name = Tstruct_file.split('.')[0]

data = np.loadtxt(data_directory + data_file, delimiter=',', skiprows=1)
Tstruct = load_struct(struct_directory + Tstruct_file)

sizes = np.linspace(start_size, end_size, n_samples, dtype=int)
list_structures = []
list_pvalues = []
for size in sizes:
    print(size)
    list_restart = []
    pvalues = []
    for i in range(n_restart):
        sample = data[np.random.randint(0, len(data), size=size)]
        sample = ot.Sample(sample)
        sample.setDescription(Tstruct.names())
        learner = otagr.ContinuousPC(sample, binNumber, alpha)
        dag = learner.learnDAG()
        list_restart.append(dag)
        nodes = dag.getDescription()
        for n in it.combinations(nodes,2):
            pvalues.append(learner.getPValue(n[0], n[1]))
    list_pvalues.append(pvalues)
    list_structures.append(list_restart)

list_pvalues = np.array(list_pvalues)

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
        #print(scores)
        list_precision.append(scores['precision'])
        list_recall.append(scores['recall'])
        list_fscore.append(scores['fscore'])
        
    precision.append(list_precision)
    recall.append(list_recall)
    fscore.append(list_fscore)

# Computing mean over the n_samples of each size
mean_precision = np.mean(precision, axis=1).reshape((n_samples,1))
mean_recall = np.mean(recall, axis=1).reshape((n_samples,1))
mean_fscore = np.mean(fscore, axis=1).reshape((n_samples,1))

# Computing standard deviation over the n_samples of each size
std_precision = np.std(precision, axis=1).reshape((n_samples,1))
std_recall = np.std(recall, axis=1).reshape((n_samples,1))
std_fscore = np.std(fscore, axis=1).reshape((n_samples,1))

# Reshaping sizes for concatenation
sizes = sizes.reshape(n_samples,1)

results = np.concatenate((sizes, mean_precision, mean_recall, mean_fscore,
                          std_precision, std_recall, std_fscore), axis=1)

header = "Size, Precision_mean, Recall_mean, Fscore_mean, " + \
         "Precision_std, Recall_std, Fscore_std"
title = "fscore_cpc_"  + data_file_name + "_" + "r" + str(n_restart) + \
        "spms" + str(binNumber) + "alpha" + str(int(100*alpha)) + \
        "s" + str(n_samples) + "f" + str(start_size) + "e" + str(end_size)

np.savetxt(data_directory + title + ".csv",
           results, fmt="%f", delimiter=',', header=header)

alpha_t = 0.1
#plot_error(sizes, mean_precision, std_precision, alpha_t)
#plot_error(sizes, mean_recall, std_recall, alpha_t)
plot_error(sizes, mean_fscore, std_fscore, alpha_t)

#plt.plot(sizes, mean_precision, label='precision')
#plt.plot(sizes, mean_recall, label='recall')
plt.plot(sizes, mean_fscore, label='fscore')


plt.legend()
plt.savefig(data_directory + title + ".pdf", transparent=True)
plt.show()
