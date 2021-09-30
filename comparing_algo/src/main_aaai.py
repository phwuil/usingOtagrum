from pipeline import Pipeline
import otagrum as otagr
from otagrum import CorrectedMutualInformation as cmi
import numpy as np
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')
mpl.rcParams.update({'font.size': 34,
                     'xtick.major.width': 3.2,
                     'xtick.major.size': 14,
                     'ytick.major.width': 3.2,
                     'ytick.major.size': 14,
                     'axes.linewidth': 2,
                     'figure.autolayout': True,
                     'legend.frameon': False,
                     'legend.borderpad': 0.001,
                     'legend.borderaxespad': 0.2,
                     'legend.handlelength': 1.1,
                     'legend.labelspacing': 0.001})

def get_domain(left, right):
    if left%10!=0 or right%10!=0:
        print("left and right must be multiples of 10!")
    if left > right:
        print("left must be less than right!")

    a = 0
    temp = left
    while (temp%10) == 0:
        temp = temp//10
        a = a + 1

    b = 0
    temp = right
    while (temp%10) == 0:
        temp = temp/10
        b = b + 1
         
    x = left
    domain = [x]
    while x < right:
        while x < 10**(a+1) and x < right:
            x = x + 10**a
            domain.append(x)
        a = a + 1
    return domain

distributions = ['gaussian', 'student', 'dirichlet']
# distributions = ['gaussian', 'dirichlet']
# structures = ['asia', 'alarm']
structures = ['alarm']

# structures = os.listdir("../data/structures/generated")
# structures.sort()
# structures = [structure.split('.')[0] for structure in structures]
# print(structures)

size_min = {'asia':100, 'alarm':100}
# size_min = {structure:2300 for structure in structures}
size_max = {'asia':10000, 'alarm':15000}
# size_max = {structure:10000 for structure in structures}
n_points = {'asia':10, 'alarm': 15}
# n_points = {structure:2 for structure in structures}
n_restart = {'asia': 10, 'alarm':5}
# n_restart = {structure:5 for structure in structures}
xlim = {'asia':4000, 'alarm':6000}
# xlim = {structure:50 for structure in structures}
ylim = {'asia': 15, 'alarm':50}
# ylim = {structure:50 for structure in structures}

correlations = np.round(np.linspace(0.8, 0.8, 1), decimals=1)

cmiic_gaussian = Pipeline('cmiic',
                          cmode=cmi.CModeTypes_Gaussian,
                          kmode=cmi.KModeTypes_Naive)
# cmiic_gaussian.setStructurePrefix("data/structures/generated/")

cmiic_bernstein = Pipeline('cmiic',
                           cmode=cmi.CModeTypes_Bernstein,
                           kmode=cmi.KModeTypes_Naive)
# cmiic_bernstein.setStructurePrefix("data/structures/generated/")

cpc = Pipeline('cpc', binNumber=5, alpha=0.05)
# cpc.setStructurePrefix("data/structures/generated/")

cbic_gaussian = Pipeline('cbic', max_parents=4, hc_restart=5,
                         cmode=cmi.CModeTypes_Gaussian)
# cbic_gaussian.setStructurePrefix("data/structures/generated/")

# cbic_bernstein = Pipeline('elidan', max_parents=4, hc_restart=5,
                          # cmode=cmi.CModeTypes_Bernstein)

dmiic = Pipeline('dmiic', dis_method='quantile', nbins=5, threshold=25)
# dmiic.setStructurePrefix("data/structures/generated/")

plot_style_bernstein = {'linewidth':4.25,
                        'linestyle':'-',
                        'color':'royalblue',
                        'label':'b-miic'}
                        # 'label':'_nolegend_'}

plot_style_gaussian = {'linewidth':4.25,
                       'linestyle':'--',
                       'color':'goldenrod',
                       'label':'g-miic'}
                       # 'label':'_nolegend_'}

plot_style_cpc = {'linewidth':4.25,
                  'linestyle':'-.',
                  'color':'maroon',
                  # 'label':'_nolegend_'}
                  'label':'cpc'}

plot_style_cbic_gaussian = {'linewidth':4.25,
                            'linestyle':(0, (1,1)),
                            'color':'olivedrab',
                            # 'label':'_nolegend_'}
                            'label':'cbic'}

# plot_style_cbic_bernstein = {'linewidth':2.,
                             # 'linestyle':'-.',
                             # 'color':'blue',
                             # 'label':'b-cbic'}

plot_style_dmiic = {'linewidth':4.25,
                            'linestyle':'-.',
                            'color':'red',
                            'label':'d-miic'}

for structure in structures:
    print('Structure :', structure)
    cmiic_gaussian.setDataStructure(structure)
    cmiic_bernstein.setDataStructure(structure)
    cpc.setDataStructure(structure)
    cbic_gaussian.setDataStructure(structure)
    # cbic_bernstein.setDataStructure(structure)
    
    cmiic_gaussian.setResultDomain(size_min[structure],
                                   size_max[structure],
                                   n_points[structure],
                                   n_restart[structure])
    cmiic_bernstein.setResultDomain(size_min[structure],
                                    size_max[structure],
                                    n_points[structure],
                                    n_restart[structure])
    cpc.setResultDomain(size_min[structure],
                        size_max[structure],
                        n_points[structure],
                        n_restart[structure])

    cbic_gaussian.setResultDomain(size_min[structure],
                                  size_max[structure],
                                  n_points[structure],
                                  n_restart[structure])

    # cbic_bernstein.setResultDomain(size_min[structure],
                                   # size_max[structure],
                                   # n_points[structure],
                                   # n_restart[structure])

    dmiic.setResultDomain(size_min[structure],
                                  size_max[structure],
                                  n_points[structure],
                                  n_restart[structure])

    for distribution in distributions:
        print('Distribution :', distribution)
        if distribution == 'gaussian' or distribution == 'student':
            for correlation in correlations:
                print('Correlation :', correlation)
                apath = os.path.join('../figures/',
                                     distribution,
                                     structure,
                                     'r'+str(correlation).replace('.', ''))
                Path(apath).mkdir(parents=True, exist_ok=True)
                cmiic_gaussian.setDataDistribution(distribution, r=correlation)
                cmiic_bernstein.setDataDistribution(distribution, r=correlation)
                cpc.setDataDistribution(distribution, r=correlation)
                cbic_gaussian.setDataDistribution(distribution, r=correlation)
                # cbic_bernstein.setDataDistribution(distribution, r=correlation)
                dmiic.setDataDistribution(distribution, r=correlation)
                
                cmiic_gaussian.generate_data()
                cmiic_bernstein.generate_data()
                cpc.generate_data()
                cbic_gaussian.generate_data()
                # cbic_bernstein.generate_data()
                dmiic.generate_data()
                
                print('cmiic gaussian', flush=True)
                cmiic_gaussian.computeStructuralScore('skelF')
                cmiic_gaussian.computeStructuralScore('hamming')
                
                print('cmiic bernstein', flush=True)
                cmiic_bernstein.computeStructuralScore('skelF')
                cmiic_bernstein.computeStructuralScore('hamming')
                
                print('cpc', flush=True)
                cpc.computeStructuralScore('skelF')
                cpc.computeStructuralScore('hamming')

                print('cbic gaussian', flush=True)
                cbic_gaussian.computeStructuralScore('skelF')
                cbic_gaussian.computeStructuralScore('hamming')

                # print('cbic bernstein')
                # cbic_bernstein.computeStructuralScore('skelF')
                # cbic_bernstein.computeStructuralScore('hamming')

                print('dmiic', flush=True)
                dmiic.computeStructuralScore('skelF')
                dmiic.computeStructuralScore('hamming')

                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], xlim[structure]])
                ax.set_ylim(0,1)
                cmiic_bernstein.plotScore('skelF', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotScore('skelF', fig, ax, **plot_style_gaussian)
                cpc.plotScore('skelF', fig, ax, **plot_style_cpc)
                cbic_gaussian.plotScore('skelF', fig, ax, **plot_style_cbic_gaussian)
                dmiic.plotScore('skelF', fig, ax, **plot_style_dmiic)
                # for side in ax.spines.keys():
                    # ax.spines[side].set_linewidth(4)
                # cbic_bernstein.plotScore('skelF', fig, ax, **plot_style_cbic_bernstein)
                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels, loc='lower right')
                plt.savefig(os.path.join(apath,
                                         '_'.join(['fscore', distribution , structure]) +'.pdf'),
                            transparent=True)
                
                
                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                ax.set_ylim(0,ylim[structure])
                cmiic_bernstein.plotScore('hamming', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotScore('hamming', fig, ax, **plot_style_gaussian)
                cpc.plotScore('hamming', fig, ax, **plot_style_cpc)
                cbic_gaussian.plotScore('hamming', fig, ax, **plot_style_cbic_gaussian)
                dmiic.plotScore('hamming', fig, ax, **plot_style_dmiic)
                # for side in ax.spines.keys():
                    # ax.spines[side].set_linewidth(4)
                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels, loc='upper right')
                # cbic_bernstein.plotScore('hamming', fig, ax, **plot_style_cbic_bernstein)
                plt.savefig(os.path.join(apath,
                                         '_'.join(['hamming', distribution , structure]) +'.pdf'),
                        transparent=True)

                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                # ax.set_ylim(0,ylim[structure])
                cmiic_bernstein.plotTime(fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotTime(fig, ax, **plot_style_gaussian)
                cpc.plotTime(fig, ax, **plot_style_cpc)
                cbic_gaussian.plotTime(fig, ax, **plot_style_cbic_gaussian)
                dmiic.plotTime(fig, ax, **plot_style_dmiic)
                # for side in ax.spines.keys():
                    # ax.spines[side].set_linewidth(4)
                # cbic_bernstein.plotScore('hamming', fig, ax, **plot_style_cbic_bernstein)
                handles, labels = ax.get_legend_handles_labels()
                handles = [h[0] for h in handles]
                ax.legend(handles, labels)
                plt.savefig(os.path.join(apath,
                                         '_'.join(['time_complexity', distribution , structure]) +'.pdf'),
                        transparent=True)
                
                
        elif distribution == 'dirichlet':
            bpath = os.path.join('../figures/', distribution, structure)
            Path(bpath).mkdir(parents=True, exist_ok=True)
            cmiic_gaussian.setDataDistribution(distribution)
            cmiic_bernstein.setDataDistribution(distribution)
            cpc.setDataDistribution(distribution)
            cbic_gaussian.setDataDistribution(distribution)
            # cbic_bernstein.setDataDistribution(distribution)
            dmiic.setDataDistribution(distribution)
            
            cmiic_gaussian.generate_data()
            cmiic_bernstein.generate_data()
            cpc.generate_data()
            cbic_gaussian.generate_data()
            # cbic_bernstein.generate_data()
            dmiic.generate_data()
            
            print('cmiic gaussian')
            cmiic_gaussian.computeStructuralScore('skelF')
            cmiic_gaussian.computeStructuralScore('hamming')
            
            print('cmiic bernstein')
            cmiic_bernstein.computeStructuralScore('skelF')
            cmiic_bernstein.computeStructuralScore('hamming')
            
            print('cpc')
            cpc.computeStructuralScore('skelF')
            cpc.computeStructuralScore('hamming')

            print('cbic gaussian')
            cbic_gaussian.computeStructuralScore('skelF')
            cbic_gaussian.computeStructuralScore('hamming')

            # print('cbic bernstein')
            # cbic_bernstein.computeStructuralScore('skelF')
            # cbic_bernstein.computeStructuralScore('hamming')

            print('dmiic')
            dmiic.computeStructuralScore('skelF')
            dmiic.computeStructuralScore('hamming')
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], xlim[structure]])
            ax.set_ylim(0,1)
            cmiic_bernstein.plotScore('skelF', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotScore('skelF', fig, ax, **plot_style_gaussian)
            cpc.plotScore('skelF', fig, ax, **plot_style_cpc)
            cbic_gaussian.plotScore('skelF', fig, ax, **plot_style_cbic_gaussian)
            dmiic.plotScore('skelF', fig, ax, **plot_style_dmiic)
            # for side in ax.spines.keys():
                # ax.spines[side].set_linewidth(4)
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            ax.legend(handles, labels, loc='lower right')
            # cbic_bernstein.plotScore('skelF', fig, ax, **plot_style_cbic_bernstein)
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['fscore', distribution , structure]) +'.pdf'),
                        transparent=True)
            
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], size_max[structure]])
            ax.set_ylim(0,ylim[structure])
            cmiic_bernstein.plotScore('hamming', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotScore('hamming', fig, ax, **plot_style_gaussian)
            cpc.plotScore('hamming', fig, ax, **plot_style_cpc)
            cbic_gaussian.plotScore('hamming', fig, ax, **plot_style_cbic_gaussian)
            dmiic.plotScore('hamming', fig, ax, **plot_style_dmiic)
            # for side in ax.spines.keys():
                # ax.spines[side].set_linewidth(4)
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            ax.legend(handles, labels, loc='upper right')
            # cbic_bernstein.plotScore('hamming', fig, ax, **plot_style_cbic_bernstein)
            print("Saving figure in {}".format(os.path.join(bpath,
                                                            '_'.join(['hamming',
                                                                      distribution,
                                                                      structure,]) + '.pdf')))
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['hamming', distribution , structure]) +'.pdf'),
                        transparent=True)

            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], size_max[structure]])
            # ax.set_ylim(0,yim[structure])
            cmiic_bernstein.plotTime(fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotTime(fig, ax, **plot_style_gaussian)
            cpc.plotTime(fig, ax, **plot_style_cpc)
            cbic_gaussian.plotTime(fig, ax, **plot_style_cbic_gaussian)
            # cbic_bernstein.plotScore('hamming', fig, ax, **plot_style_cbic_bernstein)
            dmiic.plotTime(fig, ax, **plot_style_dmiic)

            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] for h in handles]
            ax.legend(handles, labels)
            # for side in ax.spines.keys():
                # ax.spines[side].set_linewidth(4)
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['time_complexity', distribution , structure]) +'.pdf'),
                        transparent=True)
