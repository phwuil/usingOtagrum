#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os.path as path
import os
import pyAgrum as gum
import openturns as ot
import otagrum as otagr

def generate_gaussian_data(ndag, size, r=0.8):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for i in range(order.getSize()):
        d = 1 + ndag.getParents(i).getSize()
        R = ot.CorrelationMatrix(d)
        for i in range(d):
            for j in range(i):
                R[i, j] = r
        jointDistributions.append(ot.Normal([0.0]*d, [1.0]*d, R).getCopula())
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(100000)
    return sample

def load_struct(file):
    with open(file, 'r') as f:
        arcs = f.read().replace('\n', '')
    return gum.fastBN(arcs)

def write_struct(file, bn):
    struct_str = ''
    names = bn.names()
    for (head,tail) in bn.arcs():
        struct_str += names[head] + "->" + names[tail] + ';'
    with open(file, 'w') as f:
        print(struct_str, file=f)
        

# Parameters
n_sample = 20
size = 100000
r = 0.8

# Loading of data and true structure
directory = "data/gaussian/alarm/"
data_file_name = "alarm_gaussian_sample"

Tstruct_file = "alarm.txt"
Tstruct_file_name = Tstruct_file.split('.')[0]

Tstruct = load_struct(path.join(directory, Tstruct_file))
ndag=otagr.NamedDAG(Tstruct)

r_subdir = 'r' + str(r).replace('.', '')
if not path.isdir(path.join(directory, r_subdir)):
    os.mkdir(path.join(directory, r_subdir))
    
for i in range(n_sample):
    sample = generate_gaussian_data(ndag, size)
    sample.exportToCSVFile(path.join(directory, r_subdir, data_file_name) + \
                           '_' + str(i+1).zfill(2) + ".csv", ',')