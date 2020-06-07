# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heatmap as hm

def plot_error(x, mean, std, alpha=0.4, ax=None, color=None):
    x, mean, std = x.flatten(), mean.flatten(), std.flatten()
    lower, upper = mean-std, mean+std
    if ax:
        ax.fill_between(x, lower, upper, alpha=alpha, color=color)
    else:
        plt.fill_between(x, lower, upper, alpha=alpha, color=color)
        
def plotComparison(aDict, struct, rows, cols):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,6))
    mini = []
    maxi = []
    for inm, name_model in enumerate(aDict):
        temp = aDict[name_model][struct]
        mini.append([np.min(temp['AC']), np.min(temp['AC|B']), np.min(temp['ABC'])])
        maxi.append([np.max(temp['AC']), np.max(temp['AC|B']), np.max(temp['ABC'])])
    mini = np.min(mini, axis=0)
    maxi = np.max(maxi, axis=0)
        
    for inm, name_model in enumerate(aDict):
        temp = aDict[name_model][struct]
        #title = "I_" + name_info
        norm = mcolors.DivergingNorm(0., vmin=mini[0], vmax=maxi[0])
        hm.heatmap(np.array(temp['AC']), np.round(cols,decimals=1), rows, norm=norm, cmap='seismic', ax=axes[0][inm], cbarlabel='$I(A;C)$')
        norm = mcolors.DivergingNorm(0., vmin=mini[1], vmax=maxi[1])
        hm.heatmap(np.array(temp['AC|B']), np.round(cols,decimals=1), rows, norm=norm, cmap='seismic', ax=axes[1][inm], cbarlabel='$I(A;C|B)$')
        norm = mcolors.DivergingNorm(0., vmin=mini[2], vmax=maxi[2])
        hm.heatmap(np.array(temp['ABC']), np.round(cols,decimals=1), rows, norm=norm, cmap='seismic', ax=axes[2][inm], cbarlabel='$I(A;B;C)$')
        #texts = annotate_heatmap(im, valfmt="{x:.3f}")
        axes[0][inm].set_title(name_model.capitalize())
    plt.savefig('comparison_' + '_'.join([nm for nm in aDict]) + '_' + struct + '.pdf', transparent=True)
