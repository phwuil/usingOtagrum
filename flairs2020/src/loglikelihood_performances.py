#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import pyAgrum as gum

import openturns as ot
import otagrum as otagr

import score as sc
import hill_climbing as hc

import utils as ut

import os
import os.path as path

import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--method")
CLI.add_argument("--distribution")
CLI.add_argument("--structure")
CLI.add_argument("--from_size")
CLI.add_argument("--to_size")
CLI.add_argument("--n_sample")
CLI.add_argument("--test_size")
CLI.add_argument("--n_restart")
CLI.add_argument("--parameters", nargs='+', type=float)
CLI.add_argument("--correlation")

args = CLI.parse_args()


if (args.distribution == "gaussian" or args.distribution == "student") and args.correlation:
    correlation = 'r' + args.correlation.replace('.', '')
else:
    correlation = ''

method = args.method
distribution = args.distribution
structure = args.structure

from_size = int(args.from_size)
to_size = int(args.to_size)
n_samples = int(args.n_sample)
n_restart = int(args.n_restart)
test_size = int(args.test_size)


# Continuous PC parameters
if method == "cpc":
    binNumber = int(args.parameters[0])            # max size of conditional set
    alpha = float(args.parameters[1])              # Confidence threshold
    parameters = [binNumber, alpha]
    
# Elidan's learning parameters
elif method == "elidan":
    max_parents = int(args.parameters[0])     # Maximum number of parents
    n_restart_hc = int(args.parameters[1])    # Number of restart for the hill climbing
    parameters = [max_parents, n_restart_hc]

else:
    print("Wrong entry for method")


# Setting directories location and files
directory = path.join(distribution, structure, correlation)
data_directory = path.join("../data/samples/", directory)
struct_directory = "../data/structures/"

directory = path.join(directory, "loglikelihood")
res_directory = "../results/"
for d in directory.split('/'):
    if d:
        res_directory = path.join(res_directory, d)
        if not path.isdir(res_directory):
            os.mkdir(res_directory)

data_file_name = '_'.join([structure, distribution])

Tstruct_file = args.structure + ".txt"
Tstruct_file_name = args.structure

# Loading true structure
Tstruct = ut.load_struct(path.join(struct_directory, Tstruct_file))

# Looking for which size we learn
sizes = np.linspace(from_size, to_size, n_samples, dtype=int)

# Looking for all the files in the directory
files_in_directory = [f for f in os.listdir(data_directory) \
                      if path.isfile(path.join(data_directory, f))]
files_in_directory.sort()
files_in_directory = files_in_directory[:n_restart]


# Computing the likelihood
Loglikelihoods = []
for f in files_in_directory:
    print("Processing file", f)
    # Loading file f
    data = ot.Sample.ImportFromTextFile(path.join(data_directory, f), ',')
    
    list_loglikelihoods = []
    for size in sizes:
        print("    Learning with", size, "data...")
        train = data[0:size]
        test = data[-test_size:]
        
        if method=="elidan":
#            c, g, s = hc.hill_climbing(train, parameters[0], parameters[1])
#            list_loglikelihoods.append(sc.log_likelihood(test, c, g)/test.getSize())
            
            kendall_tau = train.computeKendallTau()
            pearson_r = ot.CorrelationMatrix(np.sin((np.pi/2)*kendall_tau))
            
            # Create the gaussian copula with parameters pearson_r
            # if pearson_r isn't PSD, a regularization is done
            eps = 1e-6
            done = False
            while not done:
                try:    
                    gaussian_copula = ot.NormalCopula(pearson_r)
                    done = True
                except:
                    print("Regularization")
                    for i in range(pearson_r.getDimension()):
                        for j in range(i):
                            pearson_r[i,j] /= 1 + eps
            list_loglikelihoods.append(sc.log_likelihood(test, gaussian_copula, Tstruct)/test.getSize())
            
        elif method=="cpc":
#            print("Learning structure")
#            learner = otagr.ContinuousPC(train, parameters[0], parameters[1])
#            ndag = learner.learnDAG()
#            print("Learning parameters")
#            TTest = otagr.ContinuousTTest(train, alpha)
#            jointDistributions = []        
#            for i in range(ndag.getSize()):
#                d = 1 + ndag.getParents(i).getSize()
#                if d == 1:
#                    bernsteinCopula = ot.Uniform(0.0, 1.0)
#                else:
#                    K = TTest.GetK(len(sample), d)
#                    indices = [int(n) for n in ndag.getParents(i)]
#                    indices = [i] + indices
#                    bernsteinCopula = ot.EmpiricalBernsteinCopula(sample.getMarginal(indices), K, False)
#                jointDistributions.append(bernsteinCopula)
#                
#            cbn = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
#            
#            # I need to do a loop on row of the database or there is some segfault
#            #ll = cbn.computeLogPDF(test).computeMean()[0]
#            ll = 0.
#            count = 0
#            for t in test:
#                lp = cbn.computeLogPDF(t)
#                if (np.abs(lp) < 10):
#                    ll += lp
#                    count += 1
#            ll /= count
#            list_loglikelihoods.append(ll)
            
            ndag = otagr.NamedDAG(Tstruct)
            order = ndag.getTopologicalOrder()
            TTest = otagr.ContinuousTTest(train, alpha)
            jointDistributions = []        
            for i in range(order.getSize()):
                if d == 1:
                    bernsteinCopula = ot.Uniform(0.0, 1.0)
                else:
                    K = TTest.GetK(len(sample), d)
                    indices = [int(n) for n in ndag.getParents(i)]
                    indices = [i] + indices
                    bernsteinCopula = ot.EmpiricalBernsteinCopula(sample.getMarginal(indices), K, False)
                jointDistributions.append(bernsteinCopula)
                
            #print("jD", jointDistributions)
            cbn = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
            ll = 0
            for d in test:
                #print("contribution", cbn.computeLogPDF(d))
                ll += cbn.computeLogPDF(d)
            ll /= len(test)
            list_loglikelihoods.append(ll)
            
    Loglikelihoods.append(list_loglikelihoods)

# Transposing result matrix
Loglikelihoods = np.reshape(Loglikelihoods, (n_restart, n_samples)).transpose()            

Loglikelihoods = np.array(Loglikelihoods, dtype=float)
ll_mean = np.mean(Loglikelihoods, axis=1).reshape((len(Loglikelihoods),1))
ll_std = np.std(Loglikelihoods, axis=1).reshape((len(Loglikelihoods),1))

sizes = sizes.reshape(n_samples,1)
results = np.concatenate((sizes, ll_mean, ll_std), axis=1)

if args.method == "cpc":
    title = "loglikelihood_cpc_" + data_file_name + "_" + "f" + str(from_size) + \
            "t" + str(to_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mcss" + str(binNumber) + "alpha" + str(int(100*alpha))
            
elif args.method == "elidan":
    title = "loglikelihood_elidan_" + data_file_name + "_" + "f" + str(from_size) + \
            "t" + str(to_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mp" + str(max_parents) + "hcr" + str(n_restart_hc) 

np.savetxt(path.join(res_directory, title + ".csv"), results, fmt="%f", delimiter=',')

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
#            
#            
#            
#            ndag = otagr.NamedDAG(Tstruct)
#            order = ndag.getTopologicalOrder()
#            TTest = otagr.ContinuousTTest(train, alpha)
#            jointDistributions = []        
#            for i in range(order.getSize()):
#                d = 1+ndag.getParents(i).getSize()
#                if d == 1:
#                    bernsteinCopula = ot.Uniform(0.0, 1.0)
#                else:
#                    K = TTest.GetK(len(sample), d)
#                    indices = [int(n) for n in ndag.getParents(i)]
#                    indices = [i] + indices
#                    bernsteinCopula = ot.EmpiricalBernsteinCopula(sample.getMarginal(indices), K, False)
#                jointDistributions.append(bernsteinCopula)
#                
#            cbn = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
#            ll = 0
#            for d in test:
#                ll += cbn.computeLogPDF(d)
#            ll /= len(test)
#            list_loglikelihoods.append(ll)
