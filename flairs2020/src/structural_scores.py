#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator
import openturns as ot
import otagrum as otagr
import hill_climbing as hc
import matplotlib.pyplot as plt
#import itertools as it
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


def learning(sample, method):
    if method == "cpc":
        learner = otagr.ContinuousPC(sample, binNumber, alpha)
        dag = learner.learnDAG()
        
        #nodes = dag.getDescription()
        #for n in it.combinations(nodes,2):
        #    pvalues.append(learner.getPValue(n[0], n[1]))
        
        # Tstruct used as global variable : bad !
        bn = named_dag_to_bn(dag, Tstruct.names())
    
    elif method == "elidan":
        dag = hc.hill_climbing(sample, max_parents, n_restart_hc)[1]
        bn = dag_to_bn(dag, Tstruct.names())
    else:
        print("Wrong entry for method argument !")
    
    return bn


def struct_from_one_dataset(data_file, method="cpc", start=10, end=1e4, num=10, restart=20):
    
    #list_pvalues = []
    sizes = np.linspace(start, end, num, dtype=int)
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    list_structures = []
    for size in sizes:
        print(size)
        list_restart = []
        #pvalues = []
        for i in range(restart):
            #print("restart : ", i+1)
            #sample = data[np.random.randint(0, len(data), size=size)]
            sample = data[np.random.choice(np.arange(0, len(data)),
                                           size=size,
                                           replace=False)]
            sample = ot.Sample(sample)
            sample.setDescription(Tstruct.names())
        

            bn = learning(sample, method)
            list_restart.append(bn)
            
        
        #list_pvalues.append(pvalues)
        list_structures.append(list_restart)

    #list_pvalues = np.array(list_pvalues)
    return list_structures



def struct_from_multiple_dataset(directory, method="cpc", start=10, end=1e4, num=10):
    # Looking for which size we learn
    sizes = np.linspace(start, end, num, dtype=int)
    
    # Looking for all the files in the directory
    files_in_directory = [f for f in os.listdir(directory) \
                          if path.isfile(path.join(directory, f))]
    files_in_directory.sort()
    list_structures = []
    for f in files_in_directory:
        print("Processing file", f)
        # Loading file f
        data = np.loadtxt(path.join(directory, f), delimiter=',', skiprows=1)
        list_by_size = []
        for size in sizes:
            print("    Learning with", size, "data...")
            sample = data[0:size]
            sample = ot.Sample(sample)
            sample.setDescription(Tstruct.names())
            bn = learning(sample,method)
            list_by_size.append(bn)

        list_structures.append(list_by_size)

    # Transposing result matrix
    list_structures = np.reshape(list_structures, (len(files_in_directory), num)).transpose()
    return list_structures

def structure_prospecting(structures, index):
    for s in structures[index]:
        print(s.dag())

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
    mean_precision = np.mean(precision, axis=1).reshape((len(precision),1))
    mean_recall = np.mean(recall, axis=1).reshape((len(recall),1))
    mean_fscore = np.mean(fscore, axis=1).reshape((len(fscore),1))
    return mean_precision, mean_recall, mean_fscore

def compute_stds(scores):
    # Computing standard deviation over the n_samples of each size
    precision, recall, fscore = scores
    std_precision = np.std(precision, axis=1).reshape((len(precision),1))
    std_recall = np.std(recall, axis=1).reshape((len(recall),1))
    std_fscore = np.std(fscore, axis=1).reshape((len(fscore),1))
    return std_precision, std_recall, std_fscore

# Which method is tested ("cpc" or "elidan")
method = "elidan"
# Setting test mode
mode = "unique"
# Distribution model
distribution = "student"
# Structure
structure = "asia"

# Continuous PC parameters
binNumber = 5                 # max size of conditional set
alpha = 0.05                  # Confidence threshold

# Elidan's learning parameters
max_parents = 4               # Maximum number of parents
n_restart_hc = 4             # Number of restart for the hill climbing

# Learning parameters
n_samples = 30                # Number of points of the curve
n_restart = 20                # Number of restart for each point
start_size = 1000              # Left bound of the curve
end_size = 30000              # Right bound of the curve


# Setting directories location and files
directory = path.join(distribution, structure,"r08")
data_directory = path.join("../data/samples/", directory)
struct_directory = "../data/structures/"

res_directory = "../results/"
for d in directory.split('/'):
    if d:
        res_directory = path.join(res_directory, d)
        if not path.isdir(res_directory):
            os.mkdir(res_directory)

if mode == "unique":
    data_file_name = '_'.join([structure, distribution, "sample_01"])
    data_file = data_file_name + ".csv"
elif mode == "multi":
    data_file_name = '_'.join([structure, distribution])
else:
    print("Wrong entry for mode !")

Tstruct_file = structure + ".txt"
Tstruct_file_name = structure

# Loading true structure
Tstruct = load_struct(path.join(struct_directory, Tstruct_file))


if mode == "unique":
    # Learning structures on one dataset
    list_structures = struct_from_one_dataset(path.join(data_directory, data_file),
                                              method=method,
                                              start=start_size,
                                              end=end_size,
                                              num=n_samples,
                                              restart=n_restart)
elif mode == "multi":
     # Learning structures on multiple dataset
     list_structures = struct_from_multiple_dataset(data_directory, method=method,
                                                    start=start_size,
                                                    end=end_size,
                                                    num=n_samples)
else:
    print("This mode doesn't exist !")
    
# Computing structural scores
scores = scores_from_one_dataset(Tstruct, list_structures)

# Computing mean over the n_samples of each size
mean_precision, mean_recall, mean_fscore = compute_means(scores)

# Computing standard deviation over the n_samples of each size
std_precision, std_recall, std_fscore = compute_stds(scores)

# Setting sizes for which scores are computed
sizes = np.linspace(start_size, end_size, n_samples, dtype=int)
# Reshaping sizes for concatenation
sizes = sizes.reshape(n_samples,1)

results = np.concatenate((sizes, mean_precision, mean_recall, mean_fscore,
                          std_precision, std_recall, std_fscore), axis=1)

header = "Size, Precision_mean, Recall_mean, Fscore_mean, " + \
         "Precision_std, Recall_std, Fscore_std"
         
if method == "cpc":
    title = "scores_" + mode + "_cpc_" + data_file_name + "_" + "f" + str(start_size) + \
            "t" + str(end_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mcss" + str(binNumber) + "alpha" + str(int(100*alpha))
            
elif method == "elidan":
    title = "scores_" + mode + "_elidan_" + data_file_name + "_" + "f" + str(start_size) + \
            "t" + str(end_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mp" + str(max_parents) + "hcr" + str(n_restart_hc) 
else:
    print("Wrong entry for method argument")
    
print("Writing results in ", path.join(res_directory, title))
np.savetxt(path.join(res_directory, title + ".csv"),
           results, fmt="%f", delimiter=',', header=header)
