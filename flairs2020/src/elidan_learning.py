#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold

import openturns as ot

import score as sc
import hill_climbing as hc

import os.path as path

#ot.RandomGenerator.SetSeed(42)
#np.random.seed(42)

data_set_path = "../data/samples/winequality/winequality-red.csv"
data_set_name = data_set_path.split('/')[-1].split('.')[0]

data = ot.Sample.ImportFromTextFile(data_set_path, ';')
data = (data.rank()+1)/(data.getSize()+2)
data = np.array(data)

k = 10                 # Number of fold for cross-validation
n_restart = 1          # Number of restart
stop_max_parents = 5  # Maximum number of maximum parents


kf = KFold(n_splits=k, shuffle=True)

Copulas, Graphs, Scores, Loglikelihoods = [], [], [], []
for max_parents in range(stop_max_parents+1):
    print("Loop: ", max_parents)
    list_copulas, list_graphs, list_scores, list_loglikelihoods = [], [], [], []
    for train_index, test_index in kf.split(data):
        train, test = data[train_index], data[test_index]
        c, g, s = hc.hill_climbing(ot.Sample(train), max_parents, n_restart)
        
        list_copulas.append(c)
        list_graphs.append(g)
        list_scores.append(s)
        list_loglikelihoods.append(sc.log_likelihood(ot.Sample(test), c, g)/test.shape[0])
    Copulas.append(list_copulas)
    Graphs.append(list_graphs)
    Scores.append(list_scores)
    Loglikelihoods.append(list_loglikelihoods)

Loglikelihoods = np.array(Loglikelihoods, dtype=float)
ll_mean = Loglikelihoods.mean(axis=1, keepdims=True)
ll_std = Loglikelihoods.std(axis=1, keepdims=True)

n_max_parents = np.arange(stop_max_parents+1).reshape(stop_max_parents+1,1)
results = np.concatenate((n_max_parents, ll_mean, ll_std), axis=1)

header = "k=" + str(k) + ", " + "restarts=" + str(n_restart)
title = "elidan_"  + data_set_name + "_k" + str(k) + "r" + str(n_restart) + \
        "mp" + str(stop_max_parents) + ".csv"

np.savetxt(path.join("../results/winequality/",title), results, fmt="%f", delimiter=',', header=header)