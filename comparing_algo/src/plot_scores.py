#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

import numpy as np
import os.path as path
import os

import argparse

import sys
sys.path.append('..')
from continuousMIIC import CModeType, KModeType
import utils as ut
import plotting as pl

CLI = argparse.ArgumentParser()
CLI.add_argument("--score")
CLI.add_argument("--method")
CLI.add_argument("--distribution")
CLI.add_argument("--structure")
CLI.add_argument("--from_size")
CLI.add_argument("--to_size")
CLI.add_argument("--n_sample")
CLI.add_argument("--n_restart")
CLI.add_argument("--correlation")
CLI.add_argument("--mp", type=int)
CLI.add_argument("--mcss", type=int)
CLI.add_argument("--hcr", type=int)
CLI.add_argument("--alpha", type=float)
CLI.add_argument("--cmode")
CLI.add_argument("--kmode")
args = CLI.parse_args()

if args.correlation:
    correlation = 'r' + args.correlation.replace('.', '')
else:
    correlation = ''

# Arguments
score = args.score
method = args.method
distribution = args.distribution
structure = args.structure

from_size = args.from_size
to_size = args.to_size
n_sample = args.n_sample
n_restart = args.n_restart

# Continuous PC parameters
if args.mcss or args.mp:
    mcss = int(args.mcss)         # max size of conditional set
    alpha = float(args.alpha)                 # Confidence threshold
    
# Elidan's learning parameters
if args.alpha or args.hcr:
    max_parents = int(args.mp)     # Maximum number of parents
    n_restart_hc = int(args.hcr)    # Number of restart for the hill climbing

# Continuous MIIC parameters
cmode = args.cmode
kmode = args.kmode
if cmode == "bernstein":
    cmode = CModeType.Bernstein
elif cmode == "gaussian":
    cmode = CModeType.Gaussian
    
if kmode == "nocorr":
    kmode = KModeType.NoCorr
elif kmode == "naive":
    kmode = KModeType.Naive
parameters = [cmode, kmode]

# Loading of data and true structure
directory = path.join(distribution, structure, correlation, "scores")
res_directory = path.join("../../results/", directory)

fig_directory = "../../figures/"
for d in directory.split('/'):
    if d:
        fig_directory = path.join(fig_directory, d)
        if not path.isdir(fig_directory):
            os.mkdir(fig_directory)

if method == "cpc":
    s_mcss = str(mcss)
    s_alpha = str(int(100*alpha))
    suffix = ''.join(['f', from_size, 't', to_size, 's', n_sample, 'r', n_restart,
                      'mcss', s_mcss, 'alpha', s_alpha])
    res_file_name = '_'.join([score, "scores", method, structure, distribution, suffix])
                         
elif method == "elidan":
    s_mp = str(max_parents)
    s_hcr = str(n_restart_hc)
    suffix = ''.join(['f', from_size, 't', to_size, 's', n_sample, 'r', n_restart,
                      'mp', s_mp, 'hcr', s_hcr])
    res_file_name = '_'.join([score, "scores", method, structure, distribution, suffix])
elif method == "cmiic":
    suffix = ''.join(['f', from_size, 't', to_size, 's', n_sample, 'r', n_restart,
                      args.cmode, '_', args.kmode])
    res_file_name = '_'.join([score, "scores", method, structure, distribution, suffix])
elif method == "both":
    s_mcss = str(mcss)
    s_alpha = str(int(100*alpha))
    s_mp = str(max_parents)
    s_hcr = str(n_restart_hc)
    suffix_cpc = ''.join(['f', from_size, 't', to_size, 's', n_sample, 'r', n_restart,
                      'mcss', s_mcss, 'alpha', s_alpha])
    suffix_elidan = ''.join(['f', from_size, 't', to_size, 's', n_sample, 'r', n_restart,
                      'mp', s_mp, 'hcr', s_hcr])
    res_file_name_cpc = '_'.join([score, "scores", "cpc", structure, distribution, suffix_cpc])
    res_file_name_elidan = '_'.join([score, "scores", "elidan", structure, distribution, suffix_elidan])
else:
    print("Wrong entry for method !")


if (method == "cpc") or (method == "elidan") or (method == "cmiic"):
    res_file = res_file_name + '.csv'
    res = np.loadtxt(path.join(res_directory, res_file), delimiter=',').transpose()
    sizes = res[0].astype(int)
    mean_fscore, std_fscore = res[3], res[6]
elif method == "both":
    res_file_cpc = res_file_name_cpc + '.csv'
    res_file_elidan = res_file_name_elidan + '.csv'
    
    res_cpc = np.loadtxt(path.join(res_directory, res_file_cpc), delimiter=',').transpose()
    res_elidan = np.loadtxt(path.join(res_directory, res_file_elidan), delimiter=',').transpose()
    
    sizes_cpc = res_cpc[0].astype(int)
    sizes_elidan = res_elidan[0].astype(int)
    
    mean_fscore_cpc, std_fscore_cpc = res_cpc[3], res_cpc[6]
    mean_fscore_elidan, std_fscore_elidan = res_elidan[3], res_elidan[6]

fig, ax = plt.subplots()

ax.set_xlabel('')
ax.set_ylabel('')

ax.set_xlim([int(from_size), int(to_size)])
ax.set_ylim(0,1)


alpha_t = 0.4
if method == "cpc":
    ax.plot(sizes, res[3], linestyle="-.", linewidth=1.25, color="green", label='cpc')
    pl.plot_error(sizes, mean_fscore, std_fscore, alpha_t, ax=ax, color="green")
elif method == "elidan":
    ax.plot(sizes, res[3], linestyle="--", linewidth=1.25, color="orange", label='elidan')
    pl.plot_error(sizes, mean_fscore, std_fscore, alpha_t, ax=ax, color="orange")
elif method == "cmiic":
    ax.plot(sizes, res[3], linestyle="--", linewidth=1.25, color="orange", label='cmiic')
    pl.plot_error(sizes, mean_fscore, std_fscore, alpha_t, ax=ax, color="blue")
elif method == "both":
    ax.plot(sizes_cpc, res_cpc[3], linestyle="-.", linewidth=1.25, color="green", label='cpc')
    pl.plot_error(sizes_cpc, mean_fscore_cpc, std_fscore_cpc, alpha_t, ax=ax, color="green")
    ax.plot(sizes_elidan, res_elidan[3], linestyle="--", linewidth=1.25, color="orange", label='elidan')
    pl.plot_error(sizes_elidan, mean_fscore_elidan, std_fscore_elidan, alpha_t, ax=ax, color="orange")

if (method == "cpc") or (method == "elidan") or (method == "cmiic"):
    ax.legend()
    plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
    print("Saving figure in ", path.join(fig_directory, res_file_name + ".pdf"))
elif method == "both":
    suffix = ''.join(['f', from_size, 't', to_size, 's', n_sample, 'r', n_restart,
                      'mp', s_mp, 'hcr', s_hcr, 'mcss', s_mcss, 'alpha', s_alpha])
    fig_file_name = '_'.join(['fscore_comparison', structure, distribution, suffix])
    plt.savefig(path.join(fig_directory, fig_file_name + ".pdf"), transparent=True)
    print("Saving figure in ", path.join(fig_directory, fig_file_name + ".pdf"))