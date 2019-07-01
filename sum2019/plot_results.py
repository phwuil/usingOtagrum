#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import os.path

mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

fig_path = "fig"
data_set_path = "data/winequality/elidan_winequality-red_k10r2mp4s22.csv"
data_set_name = data_set_path.split('/')[-1].split('.')[0]

data = np.loadtxt(data_set_path, delimiter=',').transpose()

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

upper = data[1] - data[2]
lower = data[1] + data[2]

ax.plot(data[0].astype(int), data[1])

#ax.fill_between(data[0], lower, upper, alpha=0.4)
plt.savefig(os.path.join(fig_path, data_set_name+".pdf"), transparent=True)