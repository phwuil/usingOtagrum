#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

import numpy as np
import os.path as path
import os

import utils as ut

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
CLI.add_argument("--correlation")
CLI.add_argument("--mp", type=int)
CLI.add_argument("--mcss", type=int)
CLI.add_argument("--hcr", type=int)
CLI.add_argument("--alpha", type=float)
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
mcss = int(args.mcss)         # max size of conditional set
alpha = float(args.alpha)                 # Confidence threshold
    
# Elidan's learning parameters
max_parents = int(args.mp)     # Maximum number of parents
n_restart_hc = int(args.hcr)    # Number of restart for the hill climbing

# Loading of data and true structure
directory = path.join(distribution, structure, correlation, "scores")
res_directory = path.join("../results/", directory)

fig_directory = "../figures/"
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


if (method == "cpc") or (method == "elidan"):
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
    


#curve, mode, method, structure, distribution, parameters = res_file_name.split('_')
#parameters = re.findall(r"\d+", parameters)

#if method == "cpc":
#    from_size, to_size, n_sample, n_restart, max_condset, alpha = parameters
#    fig_title = curve.capitalize() + " for " + method + " on " + distribution \
#                  + " data " + "generated from " + structure + " network\n" \
#                  + "Mode: " + mode + ", Restarts: " + n_restart \
#                  + ", Alpha: " + str(int(alpha)/100) \
#                  + ", MaxCondSet: " + max_condset
#elif method == "elidan":
#    from_size, to_size, n_sample, n_restart, max_parents, hc_restart = parameters
#    fig_title = curve.capitalize() + " for " + method + " on " + distribution \
#                  + " data " + "generated from " + structure + " network\n" \
#                  + "Mode: " + mode + ", Restarts: " + n_restart \
#                  + ", HCRestarts: " + hc_restart \
#                  + ", MaxParents: " + max_parents

fig, ax = plt.subplots()

ax.set_xlabel('')
ax.set_ylabel('')

ax.set_xlim([int(from_size), int(to_size)])
ax.set_ylim(0,1)

#ax.set_title(fig_title)

#ax.yaxis.set_major_formatter(ScalarFormatter())
#ax.yaxis.major.formatter._useMathText = True
#ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
#ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
#ax.yaxis.set_label_coords(0.63,1.01)
#ax.yaxis.tick_right()
#nameOfPlot = 'GDP per hour (constant prices, indexed to 2007)'
#plt.ylabel(nameOfPlot,rotation=0)
#ax.legend(frameon=False, loc='upper left',ncol=2,handlelength=4)

alpha_t = 0.4
if method == "cpc":
    ax.plot(sizes, res[3], linestyle="-.", linewidth=1.25, color="green", label='cpc')
    ut.plot_error(sizes, mean_fscore, std_fscore, alpha_t, ax=ax, color="green")
elif method == "elidan":
    ax.plot(sizes, res[3], linestyle="--", linewidth=1.25, color="orange", label='elidan')
    ut.plot_error(sizes, mean_fscore, std_fscore, alpha_t, ax=ax, color="orange")
elif method == "both":
    ax.plot(sizes_cpc, res_cpc[3], linestyle="-.", linewidth=1.25, color="green", label='cpc')
    ut.plot_error(sizes_cpc, mean_fscore_cpc, std_fscore_cpc, alpha_t, ax=ax, color="green")
    ax.plot(sizes_elidan, res_elidan[3], linestyle="--", linewidth=1.25, color="orange", label='elidan')
    ut.plot_error(sizes_elidan, mean_fscore_elidan, std_fscore_elidan, alpha_t, ax=ax, color="orange")

if (method == "cpc") or (method == "elidan"):
    ax.legend()
    plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
    print("Saving figure in ", path.join(fig_directory, res_file_name + ".pdf"))
elif method == "both":
    suffix = ''.join(['f', from_size, 't', to_size, 's', n_sample, 'r', n_restart,
                      'mp', s_mp, 'hcr', s_hcr, 'mcss', s_mcss, 'alpha', s_alpha])
    fig_file_name = '_'.join(['fscore_comparison', structure, distribution, suffix])
    plt.savefig(path.join(fig_directory, fig_file_name + ".pdf"), transparent=True)
    print("Saving figure in ", path.join(fig_directory, fig_file_name + ".pdf"))