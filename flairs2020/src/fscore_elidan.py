#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator
import openturns as ot
import hill_climbing as hc
import matplotlib.pyplot as plt
import os.path as path

def dag_to_bn(dag, names):
    # DAG to BN
    bn = gum.BayesNet()
    for name in names:
        bn.add(gum.LabelizedVariable(name))
    for arc in dag.arcs():
        bn.addArc(arc[0], arc[1])
    return bn

def load_struct(file):
    with open(file, 'r') as f:
        arcs = f.read().replace('\n', '')
    return gum.fastBN(arcs)

def plot_error(x, mean, std, alpha=0.4):
    x, mean, std = x.flatten(), mean.flatten(), std.flatten()
    lower, upper = mean-std, mean+std
    plt.fill_between(x, lower, upper, alpha=alpha)

max_parents = 4  # Maximum number of parents
n_restart_hc = 3 # Number of restart for the hill climbing
n_samples = 10   # Number of points calculated
n_restart = 20    # Number of restart for each point
start_size = 100
end_size = 10000

# Loading of data and true structure
directory = "gaussian/struct_1/r05/"
data_directory = path.join("../data/", directory)
struct_directory = path.join(data_directory, "..")
res_directory = path.join("../results/", directory)
fig_directory = path.join("../figures/", directory)

data_file = "gaussian_sample_01.csv"
data_file_name = data_file.split('.')[0]

Tstruct_file = "struct_1.txt"
Tstruct_file_name = Tstruct_file.split('.')[0]

data = np.loadtxt(data_directory + data_file, delimiter=',', skiprows=1)
Tstruct = load_struct(path.join(struct_directory, Tstruct_file))


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


# Writting and plotting figures
header = "Size, Precision_mean, Recall_mean, Fscore_mean, " + \
         "Precision_std, Recall_std, Fscore_std"
title = "fscore_elidan_"  + data_file_name + "_" + "r" + str(n_restart) + \
        "mp" + str(max_parents) + "s" + str(n_samples) + "f" + str(start_size) + \
        "e" + str(end_size)

np.savetxt(res_directory + title + ".csv",
           results, fmt="%f", delimiter=',', header=header)
 
alpha_t = 0.1
#plot_error(sizes, mean_precision, std_precision, alpha_t)
#plot_error(sizes, mean_recall, std_recall, alpha_t)
plot_error(sizes, mean_fscore, std_fscore, alpha_t)

#plt.plot(sizes, mean_precision, label='precision')
#plt.plot(sizes, mean_recall, label='recall')
plt.plot(sizes, mean_fscore, label='fscore')

plt.legend()
plt.savefig(fig_directory + title + ".pdf", transparent=True)
plt.show()