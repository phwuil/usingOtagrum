#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time

import otagrum as otagr
import pyAgrum as gum

import hill_climbing as hc
import utils as ut

import os
import os.path as path
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--distribution")
CLI.add_argument("--sample_size")
CLI.add_argument("--density")
CLI.add_argument("--from_size")
CLI.add_argument("--to_size")
CLI.add_argument("--step")
CLI.add_argument("--mp", type=int)
CLI.add_argument("--mcss", type=int)
CLI.add_argument("--hcr", type=int)
CLI.add_argument("--alpha", type=float)
CLI.add_argument("--correlation")

args = CLI.parse_args()

if (args.distribution == "gaussian" or args.distribution == "student") and args.correlation:
    correlation = 'r' + args.correlation.replace('.', '')
else:
    correlation = ''

# Continuous PC parameters
mcss = int(args.mcss)         # max size of conditional set
alpha = float(args.alpha)                 # Confidence threshold
    
# Elidan's learning parameters
max_parents = int(args.mp)     # Maximum number of parents
n_restart_hc = int(args.hcr)    # Number of restart for the hill climbing


# Learning parameters
step = int(args.step)    # Number of points of the curve
start_size = int(args.from_size)  # Left bound of the curve
end_size = int(args.to_size)      # Right bound of the curve

distribution = args.distribution
sample_size = int(args.sample_size)
density = float(args.density)

# Setting directories location and files
directory = path.join(distribution, "time_complexity", correlation)

res_directory = "../results/"
for d in directory.split('/'):
    if d:
        res_directory = path.join(res_directory, d)
        if not path.isdir(res_directory):
            os.mkdir(res_directory)


generator = gum.BNGenerator()
gum.initRandom(10)

n_nodes=[]
times_cpc=[]
times_elidan=[]
for i in range(start_size, end_size+1, step):
    print("Number of node :", i, flush=True)
    
    n_nodes.append(i)
    n_arc = int( density*(i-1) )
    bn = generator.generate(i, n_arc)
    TNdag = otagr.NamedDAG(bn.dag(), bn.names())
    
    data = ut.generate_gaussian_data(TNdag, sample_size, float(args.correlation))
    
    learner = otagr.ContinuousPC(data, mcss, alpha)
    start = time.time()
    LNdagCPC = learner.learnDAG()
    end = time.time()
    times_cpc.append(end-start)
    
    start = time.time()
    LNdagElidan = hc.hill_climbing(data, max_parents, n_restart_hc)[1]
    end = time.time()
    times_elidan.append(end - start)
    
    #LNdagCPC = [[ut.named_dag_to_bn(LNdagCPC)]]
    #LNdagElidan = [[ut.dag_to_bn(LNdagElidan, data.getDescription())]]
    
    #cpc_scores = ut.structural_scores(ut.named_dag_to_bn(TNdag), LNdagCPC)
    #elidan_scores = ut.structural_scores(ut.named_dag_to_bn(TNdag), LNdagElidan)


n_nodes = np.reshape(n_nodes, (len(n_nodes), 1))
times_cpc = np.reshape(times_cpc, (len(times_cpc), 1))
times_elidan = np.reshape(times_elidan, (len(times_elidan), 1))
results = np.concatenate((n_nodes, times_cpc, times_elidan), axis=1)

header = "N_nodes, Times_cpc, Times_elidan"
         
title = "time_" + distribution + "_f" + str(start_size) + "t" + str(end_size) + "s" + str(step) + \
        "mcss" + str(mcss) + "alpha" + str(int(100*alpha)) + \
        "mp" + str(max_parents) + "hcr" + str(n_restart_hc)
print("Writing results in ", path.join(res_directory, title + ".csv"))
np.savetxt(path.join(res_directory, title + ".csv"),
           results, fmt="%f", delimiter=',', header=header)