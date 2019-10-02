#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import os.path as path
import os
import re

import utils as ut

mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

# Loading of data and true structure
directory = "winequality"
res_file = "elidan_winequality-red_k10r5mp7.csv"
res_directory = path.join("../results/", directory)

fig_directory = "../figures/"
for d in directory.split('/'):
    if d:
        fig_directory = path.join(fig_directory, d)
        if not path.isdir(fig_directory):
            os.mkdir(fig_directory)

res_file_name = res_file.split('.')[0]

res = np.loadtxt(path.join(res_directory, res_file), delimiter=',').transpose()

max_parents = res[0].astype(int)
mean_ll, std_ll = res[1], res[2]


#from_size, to_size, n_sample, n_restart, max_condset, alpha = parameters
#fig_title = curve.capitalize() + " for " + method + " on " + distribution \
#                  + " data " + "generated from " + structure + " network\n" \
#                  + "Mode: " + mode + ", Restarts: " + n_restart \
#                  + ", Alpha: " + str(int(alpha)/100) \
#                  + ", MaxCondSet: " + max_condset
fig_title = "Wine Train"

fig, ax = plt.subplots()

ax.set_xlabel('Maximum number of parents')
ax.set_ylabel('10-fold train log-probability / instance')

ax.set_xlim([0, 4])

ax.set_ylim(0,3)

ax.set_title(fig_title)

alpha_t = 0.4

ax.plot(max_parents, mean_ll)
ut.plot_error(max_parents, mean_ll, std_ll, alpha_t, ax=ax)

plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
print("Saving figure in ", path.join(fig_directory, res_file_name + ".pdf"))
plt.show()