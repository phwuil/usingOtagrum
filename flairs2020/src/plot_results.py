#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import os.path as path
import os

def plot_error(x, mean, std, alpha=0.4, ax=None):
    x, mean, std = x.flatten(), mean.flatten(), std.flatten()
    lower, upper = mean-std, mean+std
    if ax:
        ax.fill_between(x, lower, upper, alpha=alpha)
    else:
        plt.fill_between(x, lower, upper, alpha=alpha)

mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

plot_precision = True
plot_recall = False
plot_fscore = False

# Loading of data and true structure
directory = "gaussian/struct1/r08/"
res_directory = path.join("../results", directory)

fig_directory = "../figures/"
for d in directory.split('/'):
    if d:
        fig_directory = path.join(fig_directory, d)
        if not path.isdir(fig_directory):
            os.mkdir(fig_directory)

res_file = "scores_unique_elidan_struct1_gaussian_sample_01_f100t15000s3r20mp4hcr10.csv"
res_file_name = res_file.split('.')[0]

res = np.loadtxt(res_directory + res_file, delimiter=',').transpose()

sizes = res[0].astype(int)
mean_precision, std_precision = res[1], res[4]
mean_recall, std_recall = res[2], res[5]
mean_fscore, std_fscore = res[3], res[6]


fig, ax = plt.subplots()

ax.set_xlabel('')
ax.set_ylabel('')

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
    plot_error(sizes, mean_precision, std_precision, alpha_t, ax=ax)
if plot_recall:
    ax.plot(sizes, res[2], label='recall')
    plot_error(sizes, mean_recall, std_recall, alpha_t, ax=ax) 
if plot_fscore:
    ax.plot(sizes, res[3],label='fscore')
    plot_error(sizes, mean_fscore, std_fscore, alpha_t, ax=ax)

ax.legend()
plt.savefig(path.join(fig_directory, res_file_name + ".pdf"), transparent=True)
plt.show()