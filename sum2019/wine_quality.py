#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot
import hill_climbing as hc

ot.RandomGenerator.SetSeed(42)
np.random.seed(42)

red_wine = ot.Sample.ImportFromCSVFile("data/winequality/winequality-red.csv")
red_wine = (red_wine.rank()+1)/(red_wine.getSize()+2)

G, S = hc.hill_climbing(red_wine,4,2)

print("G: ", G)
print("S: ", S)