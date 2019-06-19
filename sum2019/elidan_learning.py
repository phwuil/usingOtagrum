#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold

import openturns as ot

import score as sc
import hill_climbing as hc


#ot.RandomGenerator.SetSeed(42)
#np.random.seed(42)

data_set_path = "data/winequality/winequality-red.csv"
data_set_name = data_set_path.split('/')[-1].split('.')[0]

red_wine = ot.Sample.ImportFromCSVFile(data_set_path)
red_wine = (red_wine.rank()+1)/(red_wine.getSize()+2)
red_wine = np.array(red_wine)

k = 10
n_restart = 2
stop_max_parents = 4


kf = KFold(n_splits=k, shuffle=True)

Copulas, Graphs, Scores, Loglikelihoods = [], [], [], []
for max_parents in range(stop_max_parents+1):
    print("Loop: ", max_parents)
    list_copulas, list_graphs, list_scores, list_loglikelihoods = [], [], [], []
    for train_index, test_index in kf.split(red_wine):
        train, test = red_wine[train_index], red_wine[test_index]
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

n_max_parents = np.arange(stop_max_parents).reshape(stop_max_parents,1)
results = np.concatenate((n_max_parents, ll_mean, ll_std), axis=1)

header = "k=" + str(k) + ", " + "restarts=" + str(n_restart)
title = "elidan_"  + data_set_name + "_k" + str(k) + "r" + str(n_restart)

np.savetxt("elidan_", results, delimiter=',', header=header)