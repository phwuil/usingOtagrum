#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import utils as ut
#import itertools as it
import os.path as path
import os

import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--method")
CLI.add_argument("--mode")
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


# Continuous PC parameters
if args.method == "cpc":
    binNumber = int(args.parameters[0])                 # max size of conditional set
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

if args.mode == "unique":
    data_file_name = '_'.join([args.structure, args.distribution, "sample_01"])
    data_file = data_file_name + ".csv"
elif args.mode == "multi":
    data_file_name = '_'.join([args.structure, args.distribution])
else:
    print("Wrong entry for mode !")

Tstruct_file = args.structure + ".txt"
Tstruct_file_name = args.structure

# Loading true structure
Tstruct = ut.load_struct(path.join(struct_directory, Tstruct_file))


if args.mode == "unique":
    # Learning structures on one dataset
    list_structures = ut.struct_from_one_dataset(path.join(data_directory, data_file),
                                              method=args.method,
                                              start=start_size,
                                              end=end_size,
                                              num=n_samples,
                                              restart=n_restart)
elif args.mode == "multi":
     # Learning structures on multiple dataset
     list_structures = ut.struct_from_multiple_dataset(data_directory,
                                                       method=args.method,
                                                       parameters=parameters,
                                                       start=start_size,
                                                       end=end_size,
                                                       num=n_samples, 
                                                       restart=n_restart)
else:
    print("This mode doesn't exist !")
    
# Computing structural scores
scores = ut.structural_scores(Tstruct, list_structures)

# Computing mean over the n_samples of each size
mean_precision, mean_recall, mean_fscore = ut.compute_means(scores)

# Computing standard deviation over the n_samples of each size
std_precision, std_recall, std_fscore = ut.compute_stds(scores)

# Setting sizes for which scores are computed
sizes = np.linspace(start_size, end_size, n_samples, dtype=int)
# Reshaping sizes for concatenation
sizes = sizes.reshape(n_samples,1)

results = np.concatenate((sizes, mean_precision, mean_recall, mean_fscore,
                          std_precision, std_recall, std_fscore), axis=1)

header = "Size, Precision_mean, Recall_mean, Fscore_mean, " + \
         "Precision_std, Recall_std, Fscore_std"
         
if args.method == "cpc":
    title = "scores_" + args.mode + "_cpc_" + data_file_name + "_" + "f" + str(start_size) + \
            "t" + str(end_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mcss" + str(binNumber) + "alpha" + str(int(100*alpha))
            
elif args.method == "elidan":
    title = "scores_" + args.mode + "_elidan_" + data_file_name + "_" + "f" + str(start_size) + \
            "t" + str(end_size) + "s" + str(n_samples) + "r" + str(n_restart) + \
            "mp" + str(max_parents) + "hcr" + str(n_restart_hc) 
else:
    print("Wrong entry for method argument")
    
print("Writing results in ", path.join(res_directory, title + ".csv"))
np.savetxt(path.join(res_directory, title + ".csv"),
           results, fmt="%f", delimiter=',', header=header)

