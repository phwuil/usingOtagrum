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
directory = path.join(args.distribution, args.structure, correlation)
res_directory = path.join("../results/", directory)

fig_directory = "../figures/"
for d in directory.split('/'):
    if d:
        fig_directory = path.join(fig_directory, d)
        if not path.isdir(fig_directory):
            os.mkdir(fig_directory)

if args.method == "cpc":
    res_file_name = '_'.join(["scores",
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
    res_file_name = '_'.join(["scores",
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
mean_precision, std_precision = res[1], res[4]
mean_recall, std_recall = res[2], res[5]
mean_fscore, std_fscore = res[3], res[6]


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

ax.set_xlabel('')
ax.set_ylabel('')

ax.set_title(fig_title)
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
if plot_precision:
    ax.plot(sizes, res[1], label='precision')
    ut.plot_error(sizes, mean_precision, std_precision, alpha_t, ax=ax)
if plot_recall:
    ax.plot(sizes, res[2], label='recall')
    ut.plot_error(sizes, mean_recall, std_recall, alpha_t, ax=ax) 
if plot_fscore:
    ax.plot(sizes, res[3],label='fscore')
    ut.plot_error(sizes, mean_fscore, std_fscore, alpha_t, ax=ax)

ax.legend()
plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
print("Saving figure in ", path.join(fig_directory, res_file_name + ".pdf"))
#plt.show()