#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import utils as ut
#import itertools as it
import os.path as path
import os

import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--score")
CLI.add_argument("--method")
CLI.add_argument("--distribution")
CLI.add_argument("--structure")
CLI.add_argument("--from_size")
CLI.add_argument("--to_size")
CLI.add_argument("--n_sample")
CLI.add_argument("--n_restart")
CLI.add_argument("--parameters", nargs='+', type=float)
CLI.add_argument("--correlation")

args = CLI.parse_args()

if (args.distribution == "gaussian" or args.distribution == "student") and args.correlation:
    correlation = 'r' + args.correlation.replace('.', '')
else:
    correlation = ''

method = args.method

# Continuous PC parameters
if args.method == "cpc":
    binNumber = int(args.parameters[0])         # max size of conditional set
    alpha = args.parameters[1]                  # Confidence threshold
    parameters = [binNumber, alpha]
    
# Elidan's learning parameters
elif args.method == "elidan":
    max_parents = int(args.parameters[0])     # Maximum number of parents
    n_restart_hc = int(args.parameters[1])    # Number of restart for the hill climbing
    parameters = [max_parents, n_restart_hc]

else:
    print("Wrong entry for method")

# Learning parameters
score = args.score
n_samples = int(args.n_sample)    # Number of points of the curve
n_restart = int(args.n_restart)   # Number of restart for each point
start_size = int(args.from_size)  # Left bound of the curve
end_size = int(args.to_size)      # Right bound of the curve


# Setting directories location and files
directory = path.join(args.distribution, args.structure, correlation)
data_directory = path.join("../data/samples/", directory)
struct_directory = "../data/structures/"

directory = path.join(directory, "scores")
res_directory = "../results/"
for d in directory.split('/'):
    if d:
        res_directory = path.join(res_directory, d)
        if not path.isdir(res_directory):
            os.mkdir(res_directory)


data_file_name = '_'.join([args.structure, args.distribution])

Tstruct_file = args.structure + ".txt"
Tstruct_file_name = args.structure

# Loading true structure
Tstruct = ut.load_struct(path.join(struct_directory, Tstruct_file))

# Learning structures on multiple dataset
list_structures = ut.struct_from_multiple_dataset(data_directory,
                                                  method=args.method,
                                                  parameters=parameters,
                                                  start=start_size,
                                                  end=end_size,
                                                  num=n_samples, 
                                                  restart=n_restart)
       
    
# Setting sizes for which scores are computed
sizes = np.linspace(start_size, end_size, n_samples, dtype=int)
# Reshaping sizes for concatenation
sizes = sizes.reshape(n_samples,1)

# Computing structural scores
if (score == "skeleton") or (score == "all"):
    skel_scores = ut.structural_scores(Tstruct, list_structures, step="skeleton")
    mean_skelP, mean_skelR, mean_skelF = ut.compute_means(skel_scores)
    std_skelP, std_skelR, std_skelF = ut.compute_stds(skel_scores)
    skel_results = np.concatenate((sizes, mean_skelP, mean_skelR, mean_skelF,
                                   std_skelP, std_skelR, std_skelF), axis=1)
if (score == "dag") or (score == "all"):
    dag_scores = ut.structural_scores(Tstruct, list_structures, step="dag")
    mean_dagP, mean_dagR, mean_dagF = ut.compute_means(dag_scores)
    std_dagP, std_dagR, std_dagF = ut.compute_stds(dag_scores)
    dag_results = np.concatenate((sizes, mean_dagP, mean_dagR, mean_dagF,
                                  std_dagP, std_dagR, std_dagF), axis=1)
if (score == "hamming") or (score == "all"):
    hamming_scores = ut.hamming_score(Tstruct, list_structures)
    mean_hamming = np.mean(hamming_scores, axis=1).reshape((len(hamming_scores),1))
    std_hamming = np.std(hamming_scores, axis=1).reshape((len(hamming_scores),1))
    hamming_results = np.concatenate((sizes, mean_hamming, std_hamming), axis=1)


# Writing results
header = "Size, Precision_mean, Recall_mean, Fscore_mean, " + \
         "Precision_std, Recall_std, Fscore_std"

title = "scores_" + method + '_' + data_file_name + '_' + \
            "f" + str(start_size) + "t" + str(end_size) + "s" + str(n_samples) + \
            "r" + str(n_restart)

if args.method == "cpc":
    title += "mcss" + str(binNumber) + "alpha" + str(int(100*alpha))
elif args.method == "elidan":
    title += "mp" + str(max_parents) + "hcr" + str(n_restart_hc) 
else:
    print("Wrong entry for method argument")
    
print("Writing results in ", path.join(res_directory, title + ".csv"))

if (score == "skeleton") or (score == "all"):
    np.savetxt(path.join(res_directory, "skeleton_" + title + ".csv"),
               skel_results, fmt="%f", delimiter=',', header=header)
if (score == "dag") or (score == "all"):
    np.savetxt(path.join(res_directory, "dag_" + title + ".csv"),
               dag_results, fmt="%f", delimiter=',', header=header)
if (score == "hamming") or (score == "all"):
    np.savetxt(path.join(res_directory, "hamming_" + title + ".csv"),
               hamming_results, fmt="%f", delimiter=',')
