#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')
mpl.rcParams.update({'font.size': 16})

import numpy as np
import pandas as pd
from pathlib import Path
import os
import re

def combined_mean(m, n):
    m = np.array(m)
    n = np.array(n)
    return np.inner(m,n)/n.sum()

def combined_var(m, v, n):
    m = np.array(m)
    v = np.array(v)
    n = np.array(n)
    cm = combined_mean(m, n)
    mu2 = (m - cm)**2
    product =  np.inner(n, v + mu2)/n.sum()
    return product

def combined_std(m, v, n):
    return np.sqrt(combined_var(m, v, n))

def combined_var_df(m, v):
    n = np.full(m.shape[1], 5)
    cm = m.apply(lambda x:combined_mean(x, np.full(len(x), 5)), axis=1)
    mu2 = (np.subtract(m.T, cm)**2).T
    s = np.add(v, mu2)
    f = s.apply(lambda x:np.inner(x, n), axis=1)
    d = f/n.sum()
    return d

def combined_std_df(m, v):
    return np.sqrt(combined_var_df(m, v))

# sizes = [2]
sizes = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 62, 72, 82, 92, 102]
# sizes = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52]
distribution = "student"
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

    # cpc_time_str = 'size_' + str(size).zfill(3) + '_0*' + '/times/cpc/5_005_f2300t10000np*r5/*.csv'
    # cpc_fscore_str = 'size_' + str(size).zfill(3) + '_0*' + '/scores/skelF_cpc_5_005_f2300t10000np*r5.csv'
    # cpc_hamming_str = 'size_' + str(size).zfill(3) + '_0*' + '/scores/hamming_cpc_5_005_f2300t10000np*r5.csv'
    cpc_time_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/times/cpc/5_005_f2300t10000np*r5/*.csv'
    cpc_fscore_str = 'size_' + str(size).zfill(3) + '_0*'  + '/r08/scores/skelF_cpc_5_005_f2300t10000np*r5.csv'
    cpc_hamming_str = 'size_' + str(size).zfill(3) + '_0*'  + '/r08/scores/hamming_cpc_5_005_f2300t10000np*r5.csv'

    cpc_time_files = sorted(results_dir.joinpath(distribution).glob(cpc_time_str))
    cpc_fscore_file = sorted(results_dir.joinpath(distribution).glob(cpc_fscore_str))
    cpc_hamming_file = sorted(results_dir.joinpath(distribution).glob(cpc_hamming_str))
    print(cpc_time_files, flush=True)

    results_cpc_time = []
    for f in cpc_time_files:
        print(f, flush=True)
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        print(df)
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
    mean = cpc_fscore_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(cpc_fscore_mean_df, cpc_fscore_std_df**2)
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
    mean = cpc_hamming_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(cpc_hamming_mean_df, cpc_hamming_std_df**2)
    cpc_hamming_mean = pd.concat([cpc_hamming_mean, mean], axis=1, sort=False)
    cpc_hamming_std = pd.concat([cpc_hamming_std, std], axis=1, sort=False)

    # cbic_time_str = 'size_'+str(size).zfill(3)+ '_0*' + '/times/elidan/4_5_0_f2300t10000np*r5/*.csv'
    # cbic_fscore_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/skelF_elidan_4_5_0_f2300t10000np*r5.csv'
    # cbic_hamming_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/hamming_elidan_4_5_0_f2300t10000np*r5.csv'
    cbic_time_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/times/elidan/4_5_0_f2300t10000np*r5/*.csv'
    cbic_fscore_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/scores/skelF_elidan_4_5_0_f2300t10000np*r5.csv'
    cbic_hamming_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/scores/hamming_elidan_4_5_0_f2300t10000np*r5.csv'

    cbic_time_files = sorted(results_dir.joinpath(distribution).glob(cbic_time_str))
    cbic_fscore_file = sorted(results_dir.joinpath(distribution).glob(cbic_fscore_str))
    cbic_hamming_file = sorted(results_dir.joinpath(distribution).glob(cbic_hamming_str))
    print('CBIC time files : ')
    print(cbic_time_files, flush=True)

    results_cbic_time = []
    for f in cbic_time_files:
        print(f, flush=True)
        df = pd.read_csv(f, delimiter=',', index_col=0, header=0)
        print(df, flush=True)
        results_cbic_time.append(df)
    cbic_time_df = pd.DataFrame()
    for r in results_cbic_time:
        cbic_time_df = pd.concat([cbic_time_df, r], axis=1, sort=False)
    mean = cbic_time_df.mean(axis=1)
    std = cbic_time_df.std(axis=1)
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
    mean = cbic_fscore_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(cbic_fscore_mean_df, cbic_fscore_std_df**2)
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
    mean = cbic_hamming_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(cbic_hamming_mean_df, cbic_hamming_std_df**2)
    cbic_hamming_mean = pd.concat([cbic_hamming_mean, mean], axis=1, sort=False)
    cbic_hamming_std = pd.concat([cbic_hamming_std, std], axis=1, sort=False)


    # gmiic_time_str = 'size_'+str(size).zfill(3)+ '_0*' + '/times/cmiic/0_1_f*t*np*r5/*.csv'
    # gmiic_fscore_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/skelF_cmiic_0_1_f*t*np*r5.csv'
    # gmiic_hamming_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/hamming_cmiic_0_1_f*t*np*r5.csv'
    gmiic_time_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/times/cmiic/0_1_f100t10000np10r5/*.csv'
    gmiic_fscore_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/scores/skelF_cmiic_0_1_f*t*np*r5.csv'
    gmiic_hamming_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/scores/hamming_cmiic_0_1_f*t*np*r5.csv'

    gmiic_time_files = sorted(results_dir.joinpath(distribution).glob(gmiic_time_str))
    gmiic_fscore_file = sorted(results_dir.joinpath(distribution).glob(gmiic_fscore_str))
    gmiic_hamming_file = sorted(results_dir.joinpath(distribution).glob(gmiic_hamming_str))

    results_gmiic_time = []
    for f in gmiic_time_files:
        results_gmiic_time.append(pd.read_csv(f, delimiter=',', index_col=0, header=0))
    gmiic_time_df = pd.DataFrame()
    for r in results_gmiic_time:
        gmiic_time_df = pd.concat([gmiic_time_df, r], axis=1, sort=False)
    mean = gmiic_time_df.mean(axis=1)
    std = gmiic_time_df.std(axis=1)
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
    mean = gmiic_fscore_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(gmiic_fscore_mean_df, gmiic_fscore_std_df**2)
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
    mean = gmiic_hamming_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(gmiic_hamming_mean_df, gmiic_hamming_std_df**2)
    gmiic_hamming_mean = pd.concat([gmiic_hamming_mean, mean], axis=1, sort=False)
    gmiic_hamming_std = pd.concat([gmiic_hamming_std, std], axis=1, sort=False)


    # bmiic_time_str = 'size_'+str(size).zfill(3)+ '_0*' + '/times/cmiic/1_1_f*t*np*r5/*.csv'
    # bmiic_fscore_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/skelF_cmiic_1_1_f*t*np*r5.csv'
    # bmiic_hamming_str = 'size_'+str(size).zfill(3)+ '_0*' + '/scores/hamming_cmiic_1_1_f*t*np*r5.csv'
    bmiic_time_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/times/cmiic/1_1_f100t10000np10r5/*.csv'
    bmiic_fscore_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/scores/skelF_cmiic_1_1_f*t*np*r5.csv'
    bmiic_hamming_str = 'size_'+str(size).zfill(3) + '_0*' +'/r08/scores/hamming_cmiic_1_1_f*t*np*r5.csv'
    bmiic_time_files = sorted(results_dir.joinpath(distribution).glob(bmiic_time_str))
    bmiic_fscore_file = sorted(results_dir.joinpath(distribution).glob(bmiic_fscore_str))
    bmiic_hamming_file = sorted(results_dir.joinpath(distribution).glob(bmiic_hamming_str))

    results_bmiic_time = []
    for f in bmiic_time_files:
        results_bmiic_time.append(pd.read_csv(f, delimiter=',', index_col=0, header=0))
    bmiic_time_df = pd.DataFrame()
    for r in results_bmiic_time:
        bmiic_time_df = pd.concat([bmiic_time_df, r], axis=1, sort=False)
    mean = bmiic_time_df.mean(axis=1)
    std = bmiic_time_df.std(axis=1)
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
    mean = bmiic_fscore_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(bmiic_fscore_mean_df, bmiic_fscore_std_df**2)
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
    mean = bmiic_hamming_mean_df.apply(lambda x:combined_mean(x, np.full(len(x), 5)),
                                    axis=1)
    std = combined_std_df(bmiic_hamming_mean_df, bmiic_hamming_std_df**2)
    bmiic_hamming_mean = pd.concat([bmiic_hamming_mean, mean], axis=1, sort=False)
    bmiic_hamming_std = pd.concat([bmiic_hamming_std, std], axis=1, sort=False)
    # print("cpc fscore", cpc_fscore_std)


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

# ax.set_xlabel('Number of nodes')
# ax.set_ylabel('Time (s)')

#ax.set_title(fig_title)

# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_xlim([start_size, end_size])
# #ax.set_ylim(0.,3.)

def myLogFormat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


loglog = True
size = 10000
cpc_time_mean[size].plot(yerr=cpc_time_std[size], capsize=2, elinewidth=1.25, linestyle="-.",linewidth=1.25,color="maroon", label='cpc', ax=ax, loglog=loglog)
cbic_time_mean[size].plot(yerr=cbic_time_std[size], capsize=2, elinewidth=1.25, linestyle=(0, (1,1)),linewidth=1.25,color="olivedrab", label='cbic', ax=ax, loglog=loglog)
# gmiic_time_mean[size].plot(yerr=gmiic_time_std[size], capsize=2, elinewidth=1.25, linestyle="--",linewidth=1.25,color="goldenrod", label='g-miic', ax=ax, loglog=loglog)
# bmiic_time_mean[size].plot(yerr=bmiic_time_std[size], capsize=2, elinewidth=1.25, linestyle="-",linewidth=1.25,color="royalblue", label='b-miic', ax=ax, loglog=loglog)
ax.set_xscale('log', basex=2)
ax.set_yscale('log')
ax.set_xlim([sizes[0], sizes[-1]])
for axis in [ax.xaxis]:
    axis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
# ax.set_ylim(0.,3.)
ax.legend()

plt.savefig(fig_directory.joinpath(distribution + '/' + distribution + "_dimensional_complexity.pdf"), transparent=True)
print("Saving figure in ", fig_directory.joinpath(distribution + '/' + distribution + "_dimensional_complexity.pdf"))


loglog = False
fig_directory = Path("../figures/")
    
fig, ax = plt.subplots()

# ax.set_xlabel('Number of nodes')
# ax.set_ylabel('F-score')
# ax.set_xlim([start_size, end_size])
ax.set_ylim(0.70,1.01)

cpc_fscore_mean[size].plot(yerr=cpc_fscore_std[size], capsize=2, elinewidth=1.25, linestyle="-.",linewidth=1.25,color="maroon", label='cpc', ax=ax, loglog=loglog)
cbic_fscore_mean[size].plot(yerr=cbic_fscore_std[size], capsize=2, elinewidth=1.25, linestyle=(0, (1,1)),linewidth=1.25,color="olivedrab", label='cbic', ax=ax, loglog=loglog)
# gmiic_fscore_mean[size].plot(yerr=gmiic_fscore_std[size], capsize=2, elinewidth=1.25, linestyle="--",linewidth=1.25,color="goldenrod", label='g-miic', ax=ax, loglog=loglog)
# bmiic_fscore_mean[size].plot(yerr=bmiic_fscore_std[size], capsize=2, elinewidth=1.25, linestyle="-",linewidth=1.25,color="royalblue", label='b-miic', ax=ax, loglog=loglog)
ax.set_xlim([sizes[0], sizes[-1]])
ax.legend()

plt.savefig(fig_directory.joinpath(distribution + '/' + distribution + "_fscore_dimensional_complexity.pdf"), transparent=True)
print("Saving figure in ", fig_directory.joinpath(distribution + '/' + distribution + "_fscore_dimensional_complexity.pdf"))


fig_directory = Path("../figures/")
    
fig, ax = plt.subplots()

# ax.set_xlabel('Number of nodes')
# ax.set_ylabel('SHD')
ax.set_ylim(0,100)

cpc_hamming_mean[size].plot(yerr=cpc_hamming_std[size], capsize=2, elinewidth=1.25, linestyle="-.",linewidth=1.25,color="maroon", label='cpc', ax=ax, loglog=loglog)
cbic_hamming_mean[size].plot(yerr=cbic_hamming_std[size], capsize=2, elinewidth=1.25, linestyle=(0, (1, 1)),linewidth=1.25,color="olivedrab", label='cbic', ax=ax, loglog=loglog)
# gmiic_hamming_mean[size].plot(yerr=gmiic_hamming_std[size], capsize=2, elinewidth=1.25, linestyle="--",linewidth=1.25,color="goldenrod", label='g-miic', ax=ax, loglog=loglog)
# bmiic_hamming_mean[size].plot(yerr=bmiic_hamming_std[size], capsize=2, elinewidth=1.25, linestyle="-",linewidth=1.25,color="royalblue", label='b-miic', ax=ax, loglog=loglog)
ax.set_xlim([sizes[0], sizes[-1]])
ax.legend()

plt.savefig(fig_directory.joinpath(distribution+ '/' + distribution + "_hamming_dimensional_complexity.pdf"), transparent=True)
print("Saving figure in ", fig_directory.joinpath(distribution + '/' + distribution + "_hamming_dimensional_complexity.pdf"))
