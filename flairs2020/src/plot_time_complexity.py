#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

import numpy as np
import os.path as path
import os
import re

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

if args.correlation:
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

# Loading of data and true structure
#directory = "gaussian/asia/r08/"
directory = path.join(args.distribution, 'time_complexity', correlation)
res_directory = path.join("../results/", directory)

fig_directory = "../figures/"
for d in directory.split('/'):
    if d:
        fig_directory = path.join(fig_directory, d)
        if not path.isdir(fig_directory):
            os.mkdir(fig_directory)

res_file_name = '_'.join(["time",
                         args.distribution,
                         ''.join(['f', str(start_size),
                                  't', str(end_size),
                                  's', str(step),
                                  'mcss', str(mcss),
                                  'alpha', str(int(100*alpha)),
                                  'mp', str(max_parents),
                                  'hcr', str(n_restart_hc)])])
    
#res_file = "scores_multi_cpc_asia_gaussian_f1000t30000s8r1mcss5alpha5.csv"
res_file = res_file_name + '.csv'

res = np.loadtxt(path.join(res_directory, res_file), delimiter=',').transpose()

n_nodes = res[0].astype(int)
cpc_complexity, elidan_complexity = res[1], res[2]


fig_title = "Time complexity vs. number of nodes for both methods\n" \
              + "Alpha: " + str(alpha) + ", MaxCondSet: " + str(mcss) + ", MaxParents: " \
              + str(max_parents) + ", HCRestarts: " + str(n_restart_hc)

fig, ax = plt.subplots()

ax.set_xlabel('Number of nodes')
ax.set_ylabel('Time (s)')

#ax.set_title(fig_title)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim([start_size, end_size])
#ax.set_ylim(0.,3.)

ax.plot(n_nodes, cpc_complexity, linestyle="-.", linewidth=1.25, color="green", label='cpc')
ax.plot(n_nodes, elidan_complexity, linestyle="--", linewidth=1.25, color="orange", label='elidan')
ax.legend()

plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
print("Saving figure in ", path.join(fig_directory, res_file_name + ".pdf"))