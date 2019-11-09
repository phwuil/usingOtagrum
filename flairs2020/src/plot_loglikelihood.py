#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import os.path as path
import os
import re

import utils as ut

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
CLI.add_argument("--correlation")
CLI.add_argument("--parameters", nargs='+', type=float)

args = CLI.parse_args()

if args.correlation:
    correlation = 'r' + args.correlation.replace('.', '')
else:
    correlation = ''

mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

plot_precision = True
plot_recall = True
plot_fscore = True

# Loading of data and true structure
#directory = "gaussian/asia/r08/"
directory = path.join(args.distribution, args.structure, correlation, "loglikelihood")
res_directory = path.join("../results/", directory)

fig_directory = "../figures/"
for d in directory.split('/'):
    if d:
        fig_directory = path.join(fig_directory, d)
        if not path.isdir(fig_directory):
            os.mkdir(fig_directory)

if args.method == "cpc":
    res_file_name = '_'.join(["loglikelihood",
                         args.mode,
                         args.method,
                         args.structure,
                         args.distribution,
                         ''.join(['f', args.from_size,
                                  't', args.to_size,
                                  's', args.n_sample,
                                  'r', args.n_restart,
                                  'mcss', str(int(args.parameters[0])),
                                  'alpha', str(int(100*args.parameters[1]))])])
elif args.method == "elidan":
    res_file_name = '_'.join(["loglikelihood",
                         args.mode,
                         args.method,
                         args.structure,
                         args.distribution,
                         ''.join(['f', args.from_size,
                                  't', args.to_size,
                                  's', args.n_sample,
                                  'r', args.n_restart,
                                  'mp', str(int(args.parameters[0])),
                                  'hcr', str(int(args.parameters[1]))])])
else:
    print("Wrong entry for method !")
    
#res_file = "scores_multi_cpc_asia_gaussian_f1000t30000s8r1mcss5alpha5.csv"
res_file = res_file_name + '.csv'

res = np.loadtxt(path.join(res_directory, res_file), delimiter=',').transpose()

sizes = res[0].astype(int)
mean_ll, std_ll = res[1], res[2]


#from_size, to_size, n_sample, n_restart, max_condset, alpha = parameters
#fig_title = curve.capitalize() + " for " + method + " on " + distribution \
#                  + " data " + "generated from " + structure + " network\n" \
#                  + "Mode: " + mode + ", Restarts: " + n_restart \
#                  + ", Alpha: " + str(int(alpha)/100) \
#                  + ", MaxCondSet: " + max_condset


curve, mode, method, structure, distribution, parameters = res_file_name.split('_')
parameters = re.findall(r"\d+", parameters)

if method == "cpc":
    from_size, to_size, n_sample, n_restart, max_condset, alpha = parameters
    fig_title = curve.capitalize() + " for " + method + " on " + distribution \
                  + " data " + "generated from " + structure + " network\n" \
                  + "Mode: " + mode + ", Restarts: " + n_restart \
                  + ", Alpha: " + str(int(alpha)/100) \
                  + ", MaxCondSet: " + max_condset
elif method == "elidan":
    from_size, to_size, n_sample, n_restart, max_parents, hc_restart = parameters
    fig_title = curve.capitalize() + " for " + method + " on " + distribution \
                  + " data " + "generated from " + structure + " network\n" \
                  + "Mode: " + mode + ", Restarts: " + n_restart \
                  + ", HCRestarts: " + hc_restart \
                  + ", MaxParents: " + max_parents

fig, ax = plt.subplots()

ax.set_xlabel('Size')
ax.set_ylabel('Log-probability / instance')

ax.set_title(fig_title)

alpha_t = 0.4

ax.set_xlim([int(from_size), int(to_size)])
ax.set_ylim(0.2,0.7)

ax.plot(sizes, mean_ll)
ut.plot_error(sizes, mean_ll, std_ll, alpha_t, ax=ax)

plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
print("Saving figure in ", path.join(fig_directory, res_file_name + ".pdf"))