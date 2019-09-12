#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator
import openturns as ot
import otagrum as otagr
import hill_climbing as hc
import matplotlib.pyplot as plt
import itertools as it
import os.path as path
import os

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

def struct_from_one_dataset(data, method="cpc", start=10, end=1e4, num=10, restart=20):
    list_structures = []
    list_pvalues = []
    sizes = np.linspace(start, end, num, dtype=int)
    for size in sizes:
        print(size)
        list_restart = []
        pvalues = []
        for i in range(n_restart):
            #print("restart : ", i+1)
            sample = data[np.random.randint(0, len(data), size=size)]
            sample = ot.Sample(sample)
            sample.setDescription(Tstruct.names())
            
            if method == "cpc":
                learner = otagr.ContinuousPC(sample, binNumber, alpha)
                dag = learner.learnDAG()
                
                nodes = dag.getDescription()
                for n in it.combinations(nodes,2):
                    pvalues.append(learner.getPValue(n[0], n[1]))
                
                # Tstruct used as global variable : bad !
                bn = named_dag_to_bn(dag, Tstruct.names())
            
            elif method == "elidan":
                dag = hc.hill_climbing(sample, max_parents, n_restart_hc)[1]
                bn = dag_to_bn(dag, Tstruct.names())
            else:
                print("Wrong entry for method argument !")
            
            list_restart.append(bn)
            
        
        list_pvalues.append(pvalues)
        list_structures.append(list_restart)

    list_pvalues = np.array(list_pvalues)
    return (list_structures, list_pvalues)

def structure_prospecting(structures, index):
    for s in structures[index]:
        print(s)

def scores_from_one_dataset(true_structure, list_structures):
    precision = []
    recall = []
    fscore = []
    for l in list_structures:
        list_fscore = []
        list_recall = []
        list_precision = []
        for s in l: 
            #bn = named_dag_to_bn(s, Tstruct.names())
            comparison = GraphicalBNComparator(Tstruct, s)
            scores = comparison.scores()
            #print(scores)
            list_precision.append(scores['precision'])
            list_recall.append(scores['recall'])
            list_fscore.append(scores['fscore'])
        
        precision.append(list_precision)
        recall.append(list_recall)
        fscore.append(list_fscore)
    return precision, recall, fscore

def compute_means(scores):
    # Computing mean over the n_samples of each size
    precision, recall, fscore = scores
    mean_precision = np.mean(precision, axis=1).reshape((n_samples,1))
    mean_recall = np.mean(recall, axis=1).reshape((n_samples,1))
    mean_fscore = np.mean(fscore, axis=1).reshape((n_samples,1))
    return mean_precision, mean_recall, mean_fscore

def compute_stds(scores):
    # Computing standard deviation over the n_samples of each size
    precision, recall, fscore = scores
    std_precision = np.std(precision, axis=1).reshape((n_samples,1))
    std_recall = np.std(recall, axis=1).reshape((n_samples,1))
    std_fscore = np.std(fscore, axis=1).reshape((n_samples,1))
    return std_precision, std_recall, std_fscore

# Which method is tested ("cpc" or "elidan")
method = "cpc"

# Continuous PC parameters
binNumber = 5                 # max size of conditional set
alpha = 0.05                  # Confidence threshold

# Elidan's learning parameters
max_parents = 4               # Maximum number of parents
n_restart_hc = 3              # Number of restart for the hill climbing

# Learning parameters
n_samples = 5                # Number of points of the curve
n_restart = 10                # Number of restart for each point
start_size = 100               # Left bound of the curve
end_size = 5000              # Right bound of the curve


# Setting directories location and files
directory = "gaussian/struct_1/r03/"
data_directory = path.join("../data/samples/", directory)
struct_directory = "../data/structures/"
res_directory = path.join("../results/", directory)
fig_directory = path.join("../figures/", directory)

data_file = "struct1_gaussian_sample_01.csv"
data_file_name = data_file.split('.')[0]

Tstruct_file = "struct_1.txt"
Tstruct_file_name = Tstruct_file.split('.')[0]

# Loading data and structure
data = np.loadtxt(data_directory + data_file, delimiter=',', skiprows=1)
Tstruct = load_struct(path.join(struct_directory, Tstruct_file))

# Setting sizes for which scores are computed
sizes = np.linspace(start_size, end_size, n_samples, dtype=int)

# Learning structures on one dataset
list_structures, list_pvalues = struct_from_one_dataset(data,
                                                        method=method,
                                                        start=start_size,
                                                        end=end_size,
                                                        num=n_samples,
                                                        restart=n_restart)

# Computing structural scores
scores = scores_from_one_dataset(Tstruct, list_structures)

# Computing mean over the n_samples of each size
mean_precision, mean_recall, mean_fscore = compute_means(scores)

# Computing standard deviation over the n_samples of each size
std_precision, std_recall, std_fscore = compute_stds(scores)

# Reshaping sizes for concatenation
sizes = sizes.reshape(n_samples,1)

results = np.concatenate((sizes, mean_precision, mean_recall, mean_fscore,
                          std_precision, std_recall, std_fscore), axis=1)

header = "Size, Precision_mean, Recall_mean, Fscore_mean, " + \
         "Precision_std, Recall_std, Fscore_std"

if method == "cpc":
    title = "scores_cpc_" + data_file_name + "_" + "f" + str(start_size) + \
            "t" + str(end_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mcss" + str(binNumber) + "alpha" + str(int(100*alpha))
            
elif method == "elidan":
    title = "scores_elidan_" + data_file_name + "_" + "f" + str(start_size) + \
            "t" + str(end_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mp" + str(max_parents) + "hcr" + str(n_restart_hc) 
else:
    print("Wrong entry for method argument")
    
    
if not path.isdir(res_directory):
    os.mkdir(res_directory)
    
np.savetxt(res_directory + title + ".csv",
           results, fmt="%f", delimiter=',', header=header)
#
#alpha_t = 0.1
#
#plot_error(sizes, mean_fscore, std_fscore, alpha_t)
#plt.plot(sizes, mean_fscore, label='fscore')
#
#
#plt.legend()
#
#if not path.isdir(fig_directory):
#    os.mkdir(fig_directory)
#    
#plt.savefig(fig_directory + title + ".pdf", transparent=True)
#plt.show()
