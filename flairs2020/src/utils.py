#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pyAgrum as gum
import openturns as ot
import otagrum as otagr
import numpy as np
#import pyAgrum.lib.ipython as gnb
import hill_climbing as hc
import os.path as path
import os
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator

#def plot_pvalues(sizes, pvalues, save=False, scale=0.1):
#    x = [e for e in sizes for i in range(len(pvalues[0]))] 
#    y = pvalues.flatten()
#    plt.scatter(x, y, s=scale)
#    if save:
#        title = "pvalues_cpc_"  + data_file_name + "_" + "r" + str(n_restart) + \
#        "spms" + str(binNumber) + "alpha" + str(int(100*alpha)) + \
#        "s" + str(n_samples) + "f" + str(start_size) + "e" + str(end_size)
#        plt.savefig(data_directory + title + ".pdf", transparent=True)
#    plt.show()


def plot_error(x, mean, std, alpha=0.4, ax=None, color=None):
    x, mean, std = x.flatten(), mean.flatten(), std.flatten()
    lower, upper = mean-std, mean+std
    if ax:
        ax.fill_between(x, lower, upper, alpha=alpha, color=color)
    else:
        plt.fill_between(x, lower, upper, alpha=alpha, color=color)

def named_dag_to_bn(ndag):
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

def learning(sample, method, parameters):
    if method == "cpc":
        binNumber, alpha = parameters
        learner = otagr.ContinuousPC(sample, binNumber, alpha)
        
        ndag = learner.learnDAG()
        
        TTest = otagr.ContinuousTTest(sample, alpha)
        jointDistributions = []        
        for i in range(ndag.getSize()):
            d = 1+ndag.getParents(i).getSize()
            K = TTest.GetK(len(sample), d)
            indices = [int(n) for n in ndag.getParents(i)]
            indices = [i] + indices
            bernsteinCopula = ot.EmpiricalBernsteinCopula(sample.getMarginal(indices), K, False)
            jointDistributions.append(bernsteinCopula)
        
        bn = named_dag_to_bn(ndag)
    
    elif method == "elidan":
        #print(sample.getDescription())
        max_parents, n_restart_hc = parameters
        copula, dag = hc.hill_climbing(sample, max_parents, n_restart_hc)[0:2]
        #bn = dag_to_bn(dag, Tstruct.names())
        bn = dag_to_bn(dag, sample.getDescription())
    else:
        print("Wrong entry for method argument !")
    
    return bn


def struct_from_one_dataset(data_file, method, parameters,
                            start=10, end=1e4, num=10, restart=1):
    
    #list_pvalues = []
    sizes = np.linspace(start, end, num, dtype=int)
    
    data = ot.Sample.ImportFromTextFile(data_file, ',')
    
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
            bn = learning(sample, method)
            list_restart.append(bn)
            
        
        #list_pvalues.append(pvalues)
        list_structures.append(list_restart)

    #list_pvalues = np.array(list_pvalues)
    return list_structures



def struct_from_multiple_dataset(directory, method, parameters,
                                 start=10, end=1e4, num=10, restart=1):
    # Looking for which size we learn
    sizes = np.linspace(start, end, num, dtype=int)
    
    # Looking for all the files in the directory
    files_in_directory = [f for f in os.listdir(directory) \
                          if path.isfile(path.join(directory, f))]
    files_in_directory.sort()
    files_in_directory = files_in_directory[:restart]
    
    list_structures = []
    for f in files_in_directory:
        print("Processing file", f)
        # Loading file f
        data = ot.Sample.ImportFromTextFile(path.join(directory, f), ',')
        
        list_by_size = []
        for size in sizes:
            print("    Learning with", size, "data...", flush=True)
            sample = data[0:size]
            bn = learning(sample, method, parameters)
            list_by_size.append(bn)

        list_structures.append(list_by_size)

    # Transposing result matrix
    list_structures = np.reshape(list_structures, (len(files_in_directory), num)).transpose()
    return list_structures

def structure_prospecting(structures, index):
    for s in structures[index]:
        print(s.dag())

def structural_scores(true_structure, list_structures, step="dag"):
    precision = []
    recall = []
    fscore = []
    for l in list_structures:
        list_fscore = []
        list_recall = []
        list_precision = []
        for s in l: 
            #bn = named_dag_to_bn(s, Tstruct.names())
            comparison = GraphicalBNComparator(true_structure, s)
            if step == "skeleton":
                scores = comparison.skeletonScores()
            elif step == "dag":
                scores = comparison.scores()
            else:
                print("Wrong entry for argument!")
            
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

def generate_gaussian_data(ndag, size, r=0.8):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for k in range(order.getSize()):
        d = 1 + ndag.getParents(k).getSize()
        R = ot.CorrelationMatrix(d)
        for i in range(d):
            for j in range(i):
                R[i, j] = r
        jointDistributions.append(ot.Normal([0.0]*d, [1.0]*d, R).getCopula())
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(size)
    return sample

def generate_student_data(ndag, size, r=0.8):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for k in range(order.getSize()):
        d = 1 + ndag.getParents(k).getSize()
        R = ot.CorrelationMatrix(d)
        for i in range(d):
            for j in range(i):
                R[i, j] = r
        jointDistributions.append(ot.Student(5.0, [0.0]*d, [1.0]*d, R).getCopula())
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(size)
    return sample

def generate_dirichlet_data(ndag, size):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for k in range(order.getSize()):
        d = 1 + ndag.getParents(k).getSize()
        jointDistributions.append(ot.Dirichlet([(1.0+k)/(d+1) for k in range(d+1)]).getCopula())
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(size)
    return sample

def generate_data(ndag, size, distribution="gaussian", r=0.8):
    if distribution == "gaussian":
        sample = generate_gaussian_data(ndag, size, r)
    elif distribution == "student":
        sample = generate_student_data(ndag, size, r)
    elif distribution == "dirichlet":
        sample = generate_dirichlet_data(ndag, size)
    else:
        print("Wrong entry for the distribution !")
    return sample

def write_struct(file, bn):
    struct_str = ''
    names = bn.names()
    for (head,tail) in bn.arcs():
        struct_str += names[head] + "->" + names[tail] + ';'
    with open(file, 'w') as f:
        print(struct_str, file=f)
