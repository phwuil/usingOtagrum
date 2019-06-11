# doc
# http://openturns.github.io/openturns/master/index.html
# 
import matplotlib.pyplot as plt
import numpy as np

import pyAgrum as gum
import openturns as ot

import otagrum as otagr

R=ot.CorrelationMatrix(5)
for i in range(5):
    for j in range(i):
        R[i,j]=0.8
distribution = ot.Normal([0] * 5, [1] * 5, R)
#help(distribution)
print(distribution)


dag = gum.DAG()
for i in range(5):
  dag.addNode()
dag.addArc(0, 1)
print(dag)
