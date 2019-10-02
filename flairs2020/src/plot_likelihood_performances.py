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
directory = "gaussian/asia/r08/loglikelihood/"
res_file = "elidan_asia_gaussian_sample_01_k10r2mp4s5.csv"
res_directory = path.join("../results/", directory)

fig_directory = "../figures/"
for d in directory.split('/'):
    if d:
        fig_directory = path.join(fig_directory, d)
        if not path.isdir(fig_directory):
            os.mkdir(fig_directory)

res_file_name = res_file.split('.')[0]

res = np.loadtxt(path.join(res_directory, res_file), delimiter=',').transpose()

sizes = res[0].astype(int)
mean_ll, std_ll = res[1], res[2]


#from_size, to_size, n_sample, n_restart, max_condset, alpha = parameters
#fig_title = curve.capitalize() + " for " + method + " on " + distribution \
#                  + " data " + "generated from " + structure + " network\n" \
#                  + "Mode: " + mode + ", Restarts: " + n_restart \
#                  + ", Alpha: " + str(int(alpha)/100) \
#                  + ", MaxCondSet: " + max_condset
fig_title = "Gaussian"

fig, ax = plt.subplots()

ax.set_xlabel('Size')
ax.set_ylabel('10-fold train log-probability / instance')

ax.set_title(fig_title)

alpha_t = 0.4

ax.plot(sizes, mean_ll)
ut.plot_error(sizes, mean_ll, std_ll, alpha_t, ax=ax)

plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
print("Saving figure in ", path.join(fig_directory, res_file_name + ".pdf"))
plt.show()