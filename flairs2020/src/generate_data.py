#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    sample = copula.getSample(size)
    return sample

def generate_student_data(ndag, size, r=0.8):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for i in range(order.getSize()):
        d = 1 + ndag.getParents(i).getSize()
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
    for i in range(order.getSize()):
        d = 1 + ndag.getParents(i).getSize()
        R = ot.CorrelationMatrix(d)
        for i in range(d):
            for j in range(i):
                R[i, j] = r
        jointDistributions.append(ot.Dirichlet([(1.0+k)/(d+1) for k in range(d+1)]).getCopula())
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(size)
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

# Setting directories location and files
distribution = "gaussian"
structure = "struct1"
 
data_directory = path.join("../data/samples/", distribution)
data_file_name = structure + "_" + distribution + "_sample"

Tstruct_file = structure + ".txt"
struct_directory = "../data/structures/"

data_directory = path.join(data_directory, structure)
if not path.isdir(data_directory):
    os.mkdir(data_directory)

if distribution == "gaussian" or distribution == "student":
    r_subdir = 'r' + str(r).replace('.', '')
    data_directory = path.join(data_directory, r_subdir)
    if not path.isdir(data_directory):
        os.mkdir(data_directory)

Tstruct = load_struct(path.join(struct_directory, Tstruct_file))
#Tstruct = gum.fastBN()
ndag=otagr.NamedDAG(Tstruct)

for i in range(n_sample):
    sample = generate_gaussian_data(ndag, size)
    sample.exportToCSVFile(path.join(data_directory, data_file_name) + \
                           '_' + str(i+1).zfill(2) + ".csv", ',')