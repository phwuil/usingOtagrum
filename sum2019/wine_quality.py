#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import KFold

import openturns as ot

import score as sc
import hill_climbing as hc


#ot.RandomGenerator.SetSeed(42)
#np.random.seed(42)

red_wine = ot.Sample.ImportFromCSVFile("data/winequality/winequality-red.csv")
red_wine = (red_wine.rank()+1)/(red_wine.getSize()+2)
red_wine = np.array(red_wine)

kf = KFold(n_splits=10, shuffle=True)

C, G, S, L = [], [], [], []
for max_parents in range(5):
    print("Loop: ", max_parents)
    lc, lg, ls, ll = [], [], [], []
    for train_index, test_index in kf.split(red_wine):
        train, test = red_wine[train_index], red_wine[test_index]
        c, g, s = hc.hill_climbing(ot.Sample(train), max_parents, 5)
        
        lc.append(c)
        lg.append(g)
        ls.append(s)
        ll.append(sc.log_likelihood(ot.Sample(test), c, g)/test.shape[0])
    C.append(lc)
    G.append(lg)
    S.append(ls)
    L.append(ll)
