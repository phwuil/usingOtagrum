#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import os.path

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
plot_recall = True
plot_fscore = True

# Loading of data and true structure
data_directory = "data/gaussian/"
fig_directory = "fig/"

data_file = "fscore_cpc_gaussian_copula_sample_2_r20spms5alpha5s30f10e30000.csv"
data_file_name = data_file.split('.')[0]

data = np.loadtxt(data_directory + data_file, delimiter=',').transpose()

sizes = data[0].astype(int)
mean_precision, std_precision = data[1], data[4]
mean_recall, std_recall = data[2], data[5]
mean_fscore, std_fscore = data[3], data[6]


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

if plot_precision:
    ax.plot(sizes, data[1], label='precision')
    plot_error(sizes, mean_precision, std_precision, ax=ax)
if plot_recall:
    ax.plot(sizes, data[2], label='recall')
    plot_error(sizes, mean_recall, std_recall, ax=ax) 
if plot_fscore:
    ax.plot(sizes, data[3],label='fscore')
    plot_error(sizes, mean_fscore, std_fscore, ax=ax)

ax.legend()
#plt.savefig(os.path.join(fig_directory, data_file_name+".pdf"), transparent=True)
plt.show()