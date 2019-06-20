#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import pyAgrum as gum
from pyAgrum.lib.bn_vs_bn import GraphicalBNComparator

import openturns as ot
import otagrum as otagr

data_set_path = "data/cbn2/samplelearn.csv"
Tstruct_file_path = "data/cbn2/struct.txt"

data_set_name = data_set_path.split('/')[-1].split('.')[0]
Tstruct_file_name = Tstruct_file_path.split('/')[-1].split('.')[0]

with open(Tstruct_file_path, 'r') as file:
    arcs = file.read().replace('\n', '')
Tstruct = gum.fastBN(arcs)

data = np.loadtxt(data_set_path, delimiter=',', skiprows=1)
data = ot.Sample(data)

alpha = 0.9
binNumber = 3

learner = otagr.ContinuousPC(data, binNumber, alpha)
g2 = learner.learnDAG()

#bn = gum.BayesNet()
#for name in Tstruct.names():
#    bn.add(gum.LabelizedVariable(name))
#for arc in g.arcs():
#    bn.addArc(arc[0], arc[1])

#comparison = GraphicalBNComparator(Tstruct, bn)
#print(comparison.scores())

