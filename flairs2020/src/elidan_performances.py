#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold

import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator

import openturns as ot
import otagrum as otagr

import score as sc
import hill_climbing as hc

computeScores = False
computeLogLikelihood = True

max_parents = 4
n_restart = 2
k = 10
n_samples = 50

data_set_path = "data/sample_regis/sample.csv"

data_set_name = data_set_path.split('/')[-1].split('.')[0]

data = np.loadtxt(data_set_path, delimiter=',', skiprows=1)

if computeScores:
    
    Tstruct_file_path = "data/cbn2/struct.txt"
    Tstruct_file_name = Tstruct_file_path.split('/')[-1].split('.')[0]
    
    with open(Tstruct_file_path, 'r') as file:
        arcs = file.read().replace('\n', '')
    Tstruct = gum.fastBN(arcs)    
    
    data = ot.Sample(data)
    c, g, s = hc.hill_climbing(ot.Sample(data), max_parents, n_restart)
    
    bn = gum.BayesNet()
    for name in Tstruct.names():
        bn.add(gum.LabelizedVariable(name))
    for arc in g.arcs():
        bn.addArc(arc[0], arc[1])
    
    comparison = GraphicalBNComparator(Tstruct, bn)
    print(comparison.scores())


if computeLogLikelihood:
    sizes = np.linspace(20, 500, n_samples, dtype=int)
    Loglikelihoods = []
    Structures = []
    for size in sizes:
        print(size)
        sample = data[np.random.randint(0,len(data), size=size)]
        
        kf = KFold(n_splits=k, shuffle=True)
        
        list_loglikelihoods = []
        list_structures = []
        for train_index, test_index in kf.split(sample):
            train, test = sample[train_index], sample[test_index]
            c, g, s = hc.hill_climbing(ot.Sample(train), max_parents, n_restart)
            list_loglikelihoods.append(sc.log_likelihood(ot.Sample(test), c, g)/test.shape[0])
            list_structures.append(g)
            
        Loglikelihoods.append(list_loglikelihoods)
        Structures.append(list_structures)
    
    Loglikelihoods = np.array(Loglikelihoods, dtype=float)
    ll_mean = np.mean(Loglikelihoods, axis=1, keepdims=True)
    ll_std = Loglikelihoods.std(axis=1, keepdims=True)
    
    ll_mean = ll_mean.reshape(n_samples, 1)
    ll_std = ll_std.reshape(n_samples, 1)
    sizes = sizes.reshape(n_samples,1)
    results = np.concatenate((sizes, ll_mean, ll_std), axis=1)
    
    header = "k=" + str(k) + ", " + "restarts=" + str(n_restart)
    title = "elidan_"  + data_set_name + "_k" + str(k) + "r" + str(n_restart) + \
            "mp" + str(max_parents) + "s" + str(n_samples) + ".csv"

    np.savetxt(title, results, fmt="%f", delimiter=',', header=header)
