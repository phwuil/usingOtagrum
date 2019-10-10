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

import os.path as path


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


method = "elidan"
distribution = "dirichlet"
structure = "asia"
#r = 0.8
k = 5

from_size = 1000
to_size = 30000
n_samples = 10

if method=="cpc":
    mcss = 5
    alpha = 0.05
    parameters=[mcss, alpha]
elif method=="elidan":
    max_parents = 4
    n_restart = 4
    parameters=[max_parents, n_restart]


data_set_path = "../data/samples/dirichlet/asia/asia_dirichlet_sample_01.csv"
data_set_name = data_set_path.split('/')[-1].split('.')[0]
data = np.loadtxt(data_set_path, delimiter=',', skiprows=1)

res_dir = "../results/dirichlet/asia/loglikelihood/"

struct_directory = "../data/structures/"
Tstruct_file = structure + ".txt"
Tstruct_file_name = structure

# Loading true structure
#Tstruct = ut.load_struct(path.join(struct_directory, Tstruct_file))


Loglikelihoods = []
Structures = []
sizes = np.linspace(from_size, to_size, n_samples, dtype=int)
for size in sizes:
    print(size)
    sample = data[0:size]
    
    kf = KFold(n_splits=k, shuffle=True)
    
    list_loglikelihoods = []
    list_structures = []
    for train_index, test_index in kf.split(sample):
        train, test = sample[train_index], sample[test_index]
        if method=="elidan":
            c, g, s = hc.hill_climbing(ot.Sample(train), max_parents, n_restart)
            list_loglikelihoods.append(sc.log_likelihood(ot.Sample(test), c, g)/test.shape[0])
            list_structures.append(g)
        elif method=="cpc":
            learner = otagr.ContinuousPC(train, mcss, alpha)
            ndag = learner.learnDAG()
            order = ndag.getTopologicalOrder()
            TTest = otagr.ContinuousTTest(train, alpha)
            jointDistributions = []        
            for i in range(order.getSize()):
                d = 1+ndag.getParents(i).getSize()
                K = TTest.GetK(len(train), d)
                indices = [int(n) for n in ndag.getParents(i)]
                indices = [i] + indices
                bernsteinCopula = ot.EmpiricalBernsteinCopula(ot.Sample(train).getMarginal(indices), k, False)
                jointDistributions.append(bernsteinCopula)
                
            cbn = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
            ll = 0
            for d in test:
                ll += cbn.computeLogPDF(d)
            ll /= len(test)
            list_loglikelihoods.append(ll)
            
#        elif method == "true":
#            sample = ot.Sample(sample)
#            N = sample.getDimension()
#            # Compute the estimate of the gaussian copula    
#            kendall_tau = sample.computeKendallTau()
#            #print(kendall_tau)
#            pearson_r = ot.CorrelationMatrix(np.sin((np.pi/2)*kendall_tau))
#            
#            # Create the gaussian copula with parameters pearson_r
#            # if pearson_r isn't PSD, a regularization is done
#            eps = 1e-6
#            done = False
#            while not done:
#                try:    
#                    gaussian_copula = ot.NormalCopula(pearson_r)
#                    done = True
#                except:
#                    print("Regularization")
#                    for i in range(pearson_r.getDimension()):
#                        for j in range(i):
#                            pearson_r[i,j] /= 1 + eps
#            sample = np.array(sample)
#            list_loglikelihoods.append(sc.log_likelihood(ot.Sample(test), gaussian_copula, g)/test.shape[0])
#            list_structures.append(g)
            
            
            
#            ndag = otagr.NamedDAG(Tstruct)
#            order = ndag.getTopologicalOrder()
#            TTest = otagr.ContinuousTTest(train, alpha)
#            jointDistributions = []        
#            for i in range(order.getSize()):
#                d = 1+ndag.getParents(i).getSize()
#                K = TTest.GetK(len(train), d)
#                indices = [int(n) for n in ndag.getParents(i)]
#                indices = [i] + indices
#                bernsteinCopula = ot.EmpiricalBernsteinCopula(ot.Sample(train).getMarginal(indices), k, False)
#                jointDistributions.append(bernsteinCopula)
#                
#            cbn = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
#            ll = 0
#            for d in test:
#                ll += cbn.computeLogPDF(d)
#            ll /= len(test)
#            list_loglikelihoods.append(ll)
            
            
        
    Loglikelihoods.append(list_loglikelihoods)
    Structures.append(list_structures)

Loglikelihoods = np.array(Loglikelihoods, dtype=float)
ll_mean = np.mean(Loglikelihoods, axis=1, keepdims=True)
ll_std = Loglikelihoods.std(axis=1, keepdims=True)

ll_mean = ll_mean.reshape(n_samples, 1)
ll_std = ll_std.reshape(n_samples, 1)
sizes = sizes.reshape(n_samples,1)
results = np.concatenate((sizes, ll_mean, ll_std), axis=1)

#header = "k=" + str(k) + ", " + "restarts=" + str(n_restart)
title = "loglikelihood_kfold_cpc_trueAsia_dirichlet_f1000t30000s10r5mcss5alpha5.csv"
#title = "elidan_"  + data_set_name + "_k" + str(k) + "r" + str(n_restart) + \
#        "mp" + str(max_parents) + "s" + str(n_samples) + ".csv"

np.savetxt(path.join(res_dir, title), results, fmt="%f", delimiter=',')
