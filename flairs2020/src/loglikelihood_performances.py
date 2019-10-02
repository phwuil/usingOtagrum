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

import utils as ut


def learning(sample, method, parameters):
    if method == "cpc":
        binNumber, alpha = parameters
        learner = otagr.ContinuousPC(sample, binNumber, alpha)        
        dag = learner.learnDAG()        
        bn = ut.named_dag_to_bn(dag)
    
    elif method == "elidan":
        max_parents, n_restart_hc = parameters
        dag = hc.hill_climbing(sample, max_parents, n_restart_hc)[1]
        bn = ut.dag_to_bn(dag, sample.getDescription())
    else:
        print("Wrong entry for method argument !")
    
    return bn


method = "cpc"

from_size = 100
to_size = 500

if method=="cpc":
    mcss = 5
    alpha = 0.05
    parameters=[mcss, alpha]
elif method=="elidan":
    max_parents = 4
    n_restart = 2
    parameters[max_parents, n_restart]

k = 10

n_samples = 5

data_set_path = "../data/samples/gaussian/asia/r08/asia_gaussian_sample_01.csv"
data_set_name = data_set_path.split('/')[-1].split('.')[0]
data = np.loadtxt(data_set_path, delimiter=',', skiprows=1)


Loglikelihoods = []
Structures = []
sizes = np.linspace(from_size, to_size, n_samples, dtype=int)
for size in sizes:
    print(size)
    sample = data[np.random.choice(np.arange(0, len(data)),
                                       size=size,
                                       replace=False)]
    
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
