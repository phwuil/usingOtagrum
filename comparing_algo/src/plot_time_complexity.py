#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

import numpy as np
import pandas as pd
from pathlib import Path
import os
import re

def combined_mean(m1, n1, m2, n2):
    return (n1*m1 + n2*m2)/(n1 + n2)

def combined_var(m1, v1, n1, m2, v2, n2):
    m = combined_mean(m1, n1, m2, n2)
    return (n1*(v1 + (m1 - m)**2) + n2*(v2 + (m2 - m)**2))/(n1 + n2)

def combined_std(m1, v1, n1, m2, v2, n2):
    return np.sqrt(combined_var(m1, v1, n1, m2, v2, n2))

# sizes = [2]
sizes = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 62, 72, 82, 92, 102]
# sizes = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52]
distribution = "dirichlet"
results_dir = Path("../results")

cpc_time_mean = pd.DataFrame()
cpc_time_std = pd.DataFrame()
cpc_fscore_mean = pd.DataFrame()
cpc_fscore_std = pd.DataFrame()
cpc_hamming_mean = pd.DataFrame()
cpc_hamming_std = pd.DataFrame()

cbic_time_mean = pd.DataFrame()
cbic_time_std = pd.DataFrame()
cbic_fscore_mean = pd.DataFrame()
cbic_fscore_std = pd.DataFrame()
cbic_hamming_mean = pd.DataFrame()
cbic_hamming_std = pd.DataFrame()

gmiic_time_mean = pd.DataFrame()
gmiic_time_std = pd.DataFrame()
gmiic_fscore_mean = pd.DataFrame()
gmiic_fscore_std = pd.DataFrame()
gmiic_hamming_mean = pd.DataFrame()
gmiic_hamming_std = pd.DataFrame()

bmiic_time_mean = pd.DataFrame()
bmiic_time_std = pd.DataFrame()
bmiic_fscore_mean = pd.DataFrame()
bmiic_fscore_std = pd.DataFrame()
bmiic_hamming_mean = pd.DataFrame()
bmiic_hamming_std = pd.DataFrame()

for size in sizes:
    print(size)

    cpc_time_str = 'size_' + str(size).zfill(3) + '_0*' + '/times/cpc/5_005_f*t*np*r5/*.csv'
    # cpc_time_str = 'size_'+str(size).zfill(3)+'/r08/times/cpc/5_005_f100t10000np10r5/*.csv'
    cpc_fscore_str = 'size_' + str(size).zfill(3) + '_0*' + '/scores/skelF_cpc_5_005_f*t*np*r5.csv'
    # cpc_fscore_str = 'size_' + str(size).zfill(3) + '/r08/scores/skelF_cpc_5_005_f*t*np*r5.csv'
    cpc_hamming_str = 'size_' + str(size).zfill(3) + '_0*' + '/scores/hamming_cpc_5_005_f*t*np*r5.csv'
    # cpc_hamming_str = 'size_' + str(size).zfill(3) + '/r08/scores/hamming_cpc_5_005_f*t*np*r5.csv'

    cpc_time_files = sorted(results_dir.joinpath(distribution).glob(cpc_time_str))
    cpc_fscore_file = sorted(results_dir.joinpath(distribution).glob(cpc_fscore_str))
    cpc_hamming_file = sorted(results_dir.joinpath(distribution).glob(cpc_hamming_str))

    results_cpc_time = []
    for f in cpc_time_files:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        # print(df)
        results_cpc_time.append(df)
    cpc_time_df = pd.DataFrame()
    for r in results_cpc_time:
        cpc_time_df = pd.concat([cpc_time_df, r], axis=1, sort=False)
    mean = cpc_time_df.mean(axis=1)
    std = cpc_time_df.std(axis=1)
    cpc_time_mean = pd.concat([cpc_time_mean, mean], axis=1, sort=False)
    cpc_time_std = pd.concat([cpc_time_std, std], axis=1, sort=False)

    results_cpc_fscore_mean = []
    results_cpc_fscore_std = []
    for f in cpc_fscore_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_cpc_fscore_mean.append(df[' Mean'])
        results_cpc_fscore_std.append(df[' Std'])
    cpc_fscore_mean_df = pd.DataFrame()
    for r in results_cpc_fscore_mean:
        cpc_fscore_mean_df = pd.concat([cpc_fscore_mean_df, r], axis=1, sort=False)
    cpc_fscore_std_df = pd.DataFrame()
    for r in results_cpc_fscore_std:
        cpc_fscore_std_df = pd.concat([cpc_fscore_std_df, r], axis=1, sort=False)
    mean = cpc_fscore_mean_df.mean(axis=1)
    std = cpc_fscore_std_df.std(axis=1)
    cpc_fscore_mean = pd.concat([cpc_fscore_mean, mean], axis=1, sort=False)
    cpc_fscore_std = pd.concat([cpc_fscore_std, std], axis=1, sort=False)

    results_cpc_hamming_mean = []
    results_cpc_hamming_std = []
    for f in cpc_hamming_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_cpc_hamming_mean.append(df[' Mean'])
        results_cpc_hamming_std.append(df[' Std'])
    cpc_hamming_mean_df = pd.DataFrame()
    for r in results_cpc_hamming_mean:
        cpc_hamming_mean_df = pd.concat([cpc_hamming_mean_df, r], axis=1, sort=False)
    cpc_hamming_std_df = pd.DataFrame()
    for r in results_cpc_hamming_std:
        cpc_hamming_std_df = pd.concat([cpc_hamming_std_df, r], axis=1, sort=False)
    mean = cpc_hamming_mean_df.mean(axis=1)
    std = cpc_hamming_std_df.std(axis=1)
    cpc_hamming_mean = pd.concat([cpc_hamming_mean, mean], axis=1, sort=False)
    cpc_hamming_std = pd.concat([cpc_hamming_std, std], axis=1, sort=False)

    cbic_time_str = 'size_'+str(size).zfill(3)+ '_0*' + '/times/elidan/4_5_0_f*t*np*r5/*.csv'
    # cbic_time_str = 'size_'+str(size).zfill(3)+'/r08/times/elidan/4_5_0_f100t10000np10r5/*.csv'
    cbic_fscore_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/skelF_elidan_4_5_0_f*t*np*r5.csv'
    # cbic_fscore_str = 'size_'+str(size).zfill(3)+'/r08/scores/skelF_elidan_4_5_0_f*t*np*r5.csv'
    cbic_hamming_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/hamming_elidan_4_5_0_f*t*np*r5.csv'
    # cbic_hamming_str = 'size_'+str(size).zfill(3)+'/r08/scores/hamming_elidan_4_5_0_f*t*np*r5.csv'

    cbic_time_files = sorted(results_dir.joinpath(distribution).glob(cbic_time_str))
    cbic_fscore_file = sorted(results_dir.joinpath(distribution).glob(cbic_fscore_str))
    cbic_hamming_file = sorted(results_dir.joinpath(distribution).glob(cbic_hamming_str))

    results_cbic_time_mean = []
    results_cbic_time_std = []
    for f in cbic_time_files:
        results_cbic_time_mean.append(pd.read_csv(f, delimiter=',', index_col=0, header=0))
    cbic_time_mean_df = pd.DataFrame()
    for r in results_cbic_time_mean:
        cbic_time_mean_df = pd.concat([cbic_time_mean_df, r], axis=1, sort=False)
    cbic_time_std_df = pd.DataFrame()
    for r in results_cbic_time_std:
        cbic_time_std_df = pd.concat([cbic_time_std_df, r], axis=1, sort=False)
    mean = cbic_time_mean_df.mean(axis=1)
    std = cbic_time_std_df.std(axis=1)
    cbic_time_mean = pd.concat([cbic_time_mean, mean], axis=1, sort=False)
    cbic_time_std = pd.concat([cbic_time_std, std], axis=1, sort=False)

    results_cbic_fscore_mean = []
    results_cbic_fscore_std = []
    for f in cbic_fscore_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_cbic_fscore_mean.append(df[' Mean'])
        results_cbic_fscore_std.append(df[' Std'])
    cbic_fscore_mean_df = pd.DataFrame()
    for r in results_cbic_fscore_mean:
        cbic_fscore_mean_df = pd.concat([cbic_fscore_mean_df, r], axis=1, sort=False)
    cbic_fscore_std_df = pd.DataFrame()
    for r in results_cbic_fscore_std:
        cbic_fscore_std_df = pd.concat([cbic_fscore_std_df, r], axis=1, sort=False)
    mean = cbic_fscore_mean_df.mean(axis=1)
    std = cbic_fscore_std_df.std(axis=1)
    cbic_fscore_mean = pd.concat([cbic_fscore_mean, mean], axis=1, sort=False)
    cbic_fscore_std = pd.concat([cbic_fscore_std, std], axis=1, sort=False)

    results_cbic_hamming_mean = []
    results_cbic_hamming_std = []
    for f in cbic_hamming_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_cbic_hamming_mean.append(df[' Mean'])
        results_cbic_hamming_std.append(df[' Std'])
    cbic_hamming_mean_df = pd.DataFrame()
    for r in results_cbic_hamming_mean:
        cbic_hamming_mean_df = pd.concat([cbic_hamming_mean_df, r], axis=1, sort=False)
    cbic_hamming_std_df = pd.DataFrame()
    for r in results_cbic_hamming_std:
        cbic_hamming_std_df = pd.concat([cbic_hamming_std_df, r], axis=1, sort=False)
    mean = cbic_hamming_mean_df.mean(axis=1)
    std = cbic_hamming_std_df.std(axis=1)
    cbic_hamming_mean = pd.concat([cbic_hamming_mean, mean], axis=1, sort=False)
    cbic_hamming_std = pd.concat([cbic_hamming_std, std], axis=1, sort=False)


    gmiic_time_str = 'size_'+str(size).zfill(3)+ '_0*' + '/times/cmiic/0_1_f*t*np*r5/*.csv'
    # gmiic_time_str = 'size_'+str(size).zfill(3)+'/r08/times/cmiic/0_1_f100t10000np10r5/*.csv'
    gmiic_fscore_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/skelF_cmiic_0_1_f*t*np*r5.csv'
    # gmiic_fscore_str = 'size_'+str(size).zfill(3)+'/r08/scores/skelF_cmiic_0_1_f*t*np*r5.csv'
    gmiic_hamming_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/hamming_cmiic_0_1_f*t*np*r5.csv'
    # gmiic_hamming_str = 'size_'+str(size).zfill(3)+'/r08/scores/hamming_cmiic_0_1_f*t*np*r5.csv'

    gmiic_time_files = sorted(results_dir.joinpath(distribution).glob(gmiic_time_str))
    gmiic_fscore_file = sorted(results_dir.joinpath(distribution).glob(gmiic_fscore_str))
    gmiic_hamming_file = sorted(results_dir.joinpath(distribution).glob(gmiic_hamming_str))

    results_gmiic_time_mean = []
    results_gmiic_time_std = []
    for f in gmiic_time_files:
        results_gmiic_time_mean.append(pd.read_csv(f, delimiter=',', index_col=0, header=0))
    gmiic_time_mean_df = pd.DataFrame()
    for r in results_gmiic_time_mean:
        gmiic_time_mean_df = pd.concat([gmiic_time_mean_df, r], axis=1, sort=False)
    gmiic_time_std_df = pd.DataFrame()
    for r in results_gmiic_time_std:
        gmiic_time_std_df = pd.concat([gmiic_time_std_df, r], axis=1, sort=False)
    mean = gmiic_time_mean_df.mean(axis=1)
    std = gmiic_time_std_df.std(axis=1)
    gmiic_time_mean = pd.concat([gmiic_time_mean, mean], axis=1, sort=False)
    gmiic_time_std = pd.concat([gmiic_time_std, std], axis=1, sort=False)

    results_gmiic_fscore_mean = []
    results_gmiic_fscore_std = []
    for f in gmiic_fscore_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_gmiic_fscore_mean.append(df[' Mean'])
        results_gmiic_fscore_std.append(df[' Std'])
    gmiic_fscore_mean_df = pd.DataFrame()
    for r in results_gmiic_fscore_mean:
        gmiic_fscore_mean_df = pd.concat([gmiic_fscore_mean_df, r], axis=1, sort=False)
    gmiic_fscore_std_df = pd.DataFrame()
    for r in results_gmiic_fscore_std:
        gmiic_fscore_std_df = pd.concat([gmiic_fscore_std_df, r], axis=1, sort=False)
    mean = gmiic_fscore_mean_df.mean(axis=1)
    std = gmiic_fscore_std_df.std(axis=1)
    gmiic_fscore_mean = pd.concat([gmiic_fscore_mean, mean], axis=1, sort=False)
    gmiic_fscore_std = pd.concat([gmiic_fscore_std, std], axis=1, sort=False)

    results_gmiic_hamming_mean = []
    results_gmiic_hamming_std = []
    for f in gmiic_hamming_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_gmiic_hamming_mean.append(df[' Mean'])
        results_gmiic_hamming_std.append(df[' Std'])
    gmiic_hamming_mean_df = pd.DataFrame()
    for r in results_gmiic_hamming_mean:
        gmiic_hamming_mean_df = pd.concat([gmiic_hamming_mean_df, r], axis=1, sort=False)
    gmiic_hamming_std_df = pd.DataFrame()
    for r in results_gmiic_hamming_std:
        gmiic_hamming_std_df = pd.concat([gmiic_hamming_std_df, r], axis=1, sort=False)
    mean = gmiic_hamming_mean_df.mean(axis=1)
    std = gmiic_hamming_std_df.std(axis=1)
    gmiic_hamming_mean = pd.concat([gmiic_hamming_mean, mean], axis=1, sort=False)
    gmiic_hamming_std = pd.concat([gmiic_hamming_std, std], axis=1, sort=False)


    bmiic_time_str = 'size_'+str(size).zfill(3)+ '_0*' + '/times/cmiic/1_1_f*t*np*r5/*.csv'
    # bmiic_time_str = 'size_'+str(size).zfill(3)+'/r08/times/cmiic/1_1_f100t10000np10r5/*.csv'
    bmiic_fscore_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/skelF_cmiic_1_1_f*t*np*r5.csv'
    # bmiic_fscore_str = 'size_'+str(size).zfill(3)+'/r08/scores/skelF_cmiic_1_1_f*t*np*r5.csv'
    bmiic_hamming_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/hamming_cmiic_1_1_f*t*np*r5.csv'
    # bmiic_hamming_str = 'size_'+str(size).zfill(3)+'/r08/scores/hamming_cmiic_1_1_f*t*np*r5.csv'
    bmiic_time_files = sorted(results_dir.joinpath(distribution).glob(bmiic_time_str))
    bmiic_fscore_file = sorted(results_dir.joinpath(distribution).glob(bmiic_fscore_str))
    bmiic_hamming_file = sorted(results_dir.joinpath(distribution).glob(bmiic_hamming_str))
    results_bmiic_time_mean = []
    results_bmiic_time_std = []
    for f in bmiic_time_files:
        results_bmiic_time_mean.append(pd.read_csv(f, delimiter=',', index_col=0, header=0))
    bmiic_time_mean_df = pd.DataFrame()
    for r in results_bmiic_time_mean:
        bmiic_time_mean_df = pd.concat([bmiic_time_mean_df, r], axis=1, sort=False)
    bmiic_time_std_df = pd.DataFrame()
    for r in results_bmiic_time_std:
        bmiic_time_std_df = pd.concat([bmiic_time_std_df, r], axis=1, sort=False)
    mean = bmiic_time_mean_df.mean(axis=1)
    std = bmiic_time_std_df.std(axis=1)
    bmiic_time_mean = pd.concat([bmiic_time_mean, mean], axis=1, sort=False)
    bmiic_time_std = pd.concat([bmiic_time_std, std], axis=1, sort=False)

    results_bmiic_fscore_mean = []
    results_bmiic_fscore_std = []
    for f in bmiic_fscore_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_bmiic_fscore_mean.append(df[' Mean'])
        results_bmiic_fscore_std.append(df[' Std'])
    bmiic_fscore_mean_df = pd.DataFrame()
    for r in results_bmiic_fscore_mean:
        bmiic_fscore_mean_df = pd.concat([bmiic_fscore_mean_df, r], axis=1, sort=False)
    bmiic_fscore_std_df = pd.DataFrame()
    for r in results_bmiic_fscore_std:
        bmiic_fscore_std_df = pd.concat([bmiic_fscore_std_df, r], axis=1, sort=False)
    mean = bmiic_fscore_mean_df.mean(axis=1)
    std = bmiic_fscore_std_df.std(axis=1)
    bmiic_fscore_mean = pd.concat([bmiic_fscore_mean, mean], axis=1, sort=False)
    bmiic_fscore_std = pd.concat([bmiic_fscore_std, std], axis=1, sort=False)

    results_bmiic_hamming_mean = []
    results_bmiic_hamming_std = []
    for f in bmiic_hamming_file:
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        results_bmiic_hamming_mean.append(df[' Mean'])
        results_bmiic_hamming_std.append(df[' Std'])
    bmiic_hamming_mean_df = pd.DataFrame()
    for r in results_bmiic_hamming_mean:
        bmiic_hamming_mean_df = pd.concat([bmiic_hamming_mean_df, r], axis=1, sort=False)
    bmiic_hamming_std_df = pd.DataFrame()
    for r in results_bmiic_hamming_std:
        bmiic_hamming_std_df = pd.concat([bmiic_hamming_std_df, r], axis=1, sort=False)
    mean = bmiic_hamming_mean_df.mean(axis=1)
    std = bmiic_hamming_std_df.std(axis=1)
    bmiic_hamming_mean = pd.concat([bmiic_hamming_mean, mean], axis=1, sort=False)
    bmiic_hamming_std = pd.concat([bmiic_hamming_std, std], axis=1, sort=False)
    print("cpc fscore", cpc_fscore_mean)


cpc_time_mean.columns = sizes
cpc_time_mean = cpc_time_mean.T
cpc_time_mean.columns = cpc_time_mean.columns.astype(int)

cpc_time_std.columns = sizes
cpc_time_std = cpc_time_std.T
cpc_time_std.columns = cpc_time_std.columns.astype(int)

cpc_fscore_mean.columns = sizes
cpc_fscore_mean = cpc_fscore_mean.T
cpc_fscore_mean.columns = cpc_fscore_mean.columns.astype(int)

cpc_fscore_std.columns = sizes
cpc_fscore_std = cpc_fscore_std.T
cpc_fscore_std.columns = cpc_fscore_std.columns.astype(int)

cpc_hamming_mean.columns = sizes
cpc_hamming_mean = cpc_hamming_mean.T
cpc_hamming_mean.columns = cpc_hamming_mean.columns.astype(int)

cpc_hamming_std.columns = sizes
cpc_hamming_std = cpc_hamming_std.T
cpc_hamming_std.columns = cpc_hamming_std.columns.astype(int)

cbic_time_mean.columns = sizes
cbic_time_mean = cbic_time_mean.T
cbic_time_mean.columns = cbic_time_mean.columns.astype(int)

cbic_time_std.columns = sizes
cbic_time_std = cbic_time_std.T
cbic_time_std.columns = cbic_time_std.columns.astype(int)

cbic_fscore_mean.columns = sizes
cbic_fscore_mean = cbic_fscore_mean.T
cbic_fscore_mean.columns = cbic_fscore_mean.columns.astype(int)

cbic_fscore_std.columns = sizes
cbic_fscore_std = cbic_fscore_std.T
cbic_fscore_std.columns = cbic_fscore_std.columns.astype(int)

cbic_hamming_mean.columns = sizes
cbic_hamming_mean = cbic_hamming_mean.T
cbic_hamming_mean.columns = cbic_hamming_mean.columns.astype(int)

cbic_hamming_std.columns = sizes
cbic_hamming_std = cbic_hamming_std.T
cbic_hamming_std.columns = cbic_hamming_std.columns.astype(int)

gmiic_time_mean.columns = sizes
gmiic_time_mean = gmiic_time_mean.T
gmiic_time_mean.columns = gmiic_time_mean.columns.astype(int)

gmiic_time_std.columns = sizes
gmiic_time_std = gmiic_time_std.T
gmiic_time_std.columns = gmiic_time_std.columns.astype(int)

gmiic_fscore_mean.columns = sizes
gmiic_fscore_mean = gmiic_fscore_mean.T
gmiic_fscore_mean.columns = gmiic_fscore_mean.columns.astype(int)

gmiic_fscore_std.columns = sizes
gmiic_fscore_std = gmiic_fscore_std.T
gmiic_fscore_std.columns = gmiic_fscore_std.columns.astype(int)

gmiic_hamming_mean.columns = sizes
gmiic_hamming_mean = gmiic_hamming_mean.T
gmiic_hamming_mean.columns = gmiic_hamming_mean.columns.astype(int)

gmiic_hamming_std.columns = sizes
gmiic_hamming_std = gmiic_hamming_std.T
gmiic_hamming_std.columns = gmiic_hamming_std.columns.astype(int)

bmiic_time_mean.columns = sizes
bmiic_time_mean = bmiic_time_mean.T
bmiic_time_mean.columns = bmiic_time_mean.columns.astype(int)

bmiic_time_std.columns = sizes
bmiic_time_std = bmiic_time_std.T
bmiic_time_std.columns = bmiic_time_std.columns.astype(int)

bmiic_fscore_mean.columns = sizes
bmiic_fscore_mean = bmiic_fscore_mean.T
bmiic_fscore_mean.columns = bmiic_fscore_mean.columns.astype(int)

bmiic_fscore_std.columns = sizes
bmiic_fscore_std = bmiic_fscore_std.T
bmiic_fscore_std.columns = bmiic_fscore_std.columns.astype(int)

bmiic_hamming_mean.columns = sizes
bmiic_hamming_mean = bmiic_hamming_mean.T
bmiic_hamming_mean.columns = bmiic_hamming_mean.columns.astype(int)

bmiic_hamming_std.columns = sizes
bmiic_hamming_std = bmiic_hamming_std.T
bmiic_hamming_std.columns = bmiic_hamming_std.columns.astype(int)


fig_directory = Path("../figures/")
    
fig, ax = plt.subplots()

ax.set_xlabel('Number of nodes')
ax.set_ylabel('Time (s)')

#ax.set_title(fig_title)

# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_xlim([start_size, end_size])
# #ax.set_ylim(0.,3.)

loglog = False
size = 10000
cpc_time_mean[size].plot(linestyle="-.",linewidth=1.25,color="green", label='cpc', ax=ax, loglog=loglog)
cbic_time_mean[size].plot(linestyle="--",linewidth=1.25,color="orange", label='cbic', ax=ax, loglog=loglog)
gmiic_time_mean[size].plot(linestyle="--",linewidth=1.25,color="red", label='g-miic', ax=ax, loglog=loglog)
bmiic_time_mean[size].plot(linestyle="--",linewidth=1.25,color="blue", label='b-miic', ax=ax, loglog=loglog)
ax.legend()

plt.savefig(fig_directory.joinpath("dimensional_complexity_02.pdf"), transparent=True)
print("Saving figure in ", fig_directory.joinpath("dimensional_complexity.pdf"))


fig_directory = Path("../figures/")
    
fig, ax = plt.subplots()

ax.set_xlabel('Number of nodes')
ax.set_ylabel('F-score')

cpc_fscore_mean[size].plot(linestyle="-.",linewidth=1.25,color="green", label='cpc', ax=ax, loglog=loglog)
print("FScore cpc", cpc_fscore_std)
cbic_fscore_mean[size].plot(linestyle="--",linewidth=1.25,color="orange", label='cbic', ax=ax, loglog=loglog)
print("FScore cbic", cbic_fscore_std)
gmiic_fscore_mean[size].plot(linestyle="--",linewidth=1.25,color="red", label='g-miic', ax=ax, loglog=loglog)
print("FScore gmiic", gmiic_fscore_std)
bmiic_fscore_mean[size].plot(linestyle="--",linewidth=1.25,color="blue", label='b-miic', ax=ax, loglog=loglog)
print("FScore bmiic", bmiic_fscore_std)
ax.legend()

plt.savefig(fig_directory.joinpath("fscore_dimensional_complexity_02.pdf"), transparent=True)
print("Saving figure in ", fig_directory.joinpath("fscore_dimensional_complexity.pdf"))


fig_directory = Path("../figures/")
    
fig, ax = plt.subplots()

ax.set_xlabel('Number of nodes')
ax.set_ylabel('SHD')

cpc_hamming_mean[size].plot(linestyle="-.",linewidth=1.25,color="green", label='cpc', ax=ax, loglog=loglog)
cbic_hamming_mean[size].plot(linestyle="--",linewidth=1.25,color="orange", label='cbic', ax=ax, loglog=loglog)
gmiic_hamming_mean[size].plot(linestyle="--",linewidth=1.25,color="red", label='g-miic', ax=ax, loglog=loglog)
bmiic_hamming_mean[size].plot(linestyle="--",linewidth=1.25,color="blue", label='b-miic', ax=ax, loglog=loglog)
ax.legend()

plt.savefig(fig_directory.joinpath("hamming_dimensional_complexity_02.pdf"), transparent=True)
print("Saving figure in ", fig_directory.joinpath("hamming_dimensional_complexity.pdf"))
