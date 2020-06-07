from pipeline import Pipeline
from cmiic.continuousMIIC import CModeType, KModeType
import numpy as np
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

distributions = ['gaussian', 'student', 'dirichlet']
distributions = ['dirichlet']
structures = ['asia', 'alarm']

size_min = {'asia':100, 'alarm':100}
size_max = {'asia':300, 'alarm':300}
n_points = {'asia':3, 'alarm': 3}
n_restart = {'asia': 1, 'alarm':1}
ylim = {'asia': 10, 'alarm':50}

correlations = np.round(np.linspace(0.1, 0.9, 2), decimals=1)

# c3off2_bernstein = Pipeline('3off2', cmode=CModeType.Bernstein, kmode=KModeType)
cmiic_gaussian = Pipeline('cmiic', cmode=CModeType.Gaussian, kmode=KModeType.Naive)
cmiic_bernstein = Pipeline('cmiic', cmode=CModeType.Bernstein, kmode=KModeType.Naive)
cpc = Pipeline('cpc', binNumber=5, alpha=0.05)

# plot_style_3off2 = {'linewidth':1.25, 'linestyle':'-.', 'color':'yellow', 'label':'3off2'}
plot_style_bernstein = {'linewidth':1.25, 'linestyle':'-.', 'color':'green', 'label':'bernstein'}
plot_style_gaussian = {'linewidth':1.25, 'linestyle':'--', 'color':'orange', 'label':'gaussian'}
plot_style_cpc = {'linewidth':1.25, 'linestyle':'--', 'color':'red', 'label':'cpc'}

for structure in structures:
    print('Structure :', structure)
    # c3off2_bernstein.setDataStructure(structure)
    cmiic_gaussian.setDataStructure(structure)
    cmiic_bernstein.setDataStructure(structure)
    cpc.setDataStructure(structure)
    
    # c3off2_bernstein.setResultDomain(size_min[structure],
                                    # size_max[structure],
                                    # n_points[structure],
                                    # n_restart[structure])
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
                # c3off2_bernstein.setDataDistribution(distribution, r=correlation)
                cmiic_gaussian.setDataDistribution(distribution, r=correlation)
                cmiic_bernstein.setDataDistribution(distribution, r=correlation)
                cpc.setDataDistribution(distribution, r=correlation)
                
                # c3off2_bernstein.generate_data()
                cmiic_gaussian.generate_data()
                cmiic_bernstein.generate_data()
                cpc.generate_data()
                
                # print('c3off2 bernstein')
                # c3off2_bernstein.computeStructuralScore('skelF')
                # c3off2_bernstein.computeStructuralScore('hamming')
                # c3off2_bernstein.computeStructuralScore('dagF')
                
                print('cmiic gaussian')
                cmiic_gaussian.computeStructuralScore('skelF')
                cmiic_gaussian.computeStructuralScore('hamming')
                # cmiic_gaussian.computeStructuralScore('dagF')
                
                print('cmiic bernstein')
                cmiic_bernstein.computeStructuralScore('skelF')
                cmiic_bernstein.computeStructuralScore('hamming')
                # cmiic_bernstein.computeStructuralScore('dagF')
                
                print('cpc')
                cpc.computeStructuralScore('skelF')
                cpc.computeStructuralScore('hamming')
                # cpc.computeStructuralScore('dagF')
                
                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                ax.set_ylim(0,1)
                # c3off2_bernstein.plotScore('skelF', fig, ax, **plot_style_3off2)
                cmiic_bernstein.plotScore('skelF', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotScore('skelF', fig, ax, **plot_style_gaussian)
                cpc.plotScore('skelF', fig, ax, **plot_style_cpc)
                # elidan.plotScore('skeleton', fig, ax, **plot_style_cpc)
                plt.savefig(os.path.join(apath,'bernstein_gaussian_cpc_skelF.pdf'),
                                         transparent=True)
                
                
                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                ax.set_ylim(0,ylim[structure])
                # c3off2_bernstein.plotScore('hamming', fig, ax, **plot_style_3off2)
                cmiic_bernstein.plotScore('hamming', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotScore('hamming', fig, ax, **plot_style_gaussian)
                cpc.plotScore('hamming', fig, ax, **plot_style_cpc)
                # elidan.plotScore('hamming', fig, ax, **plot_style_cpc)
                plt.savefig(os.path.join(apath, 'bernstein_gaussian_cpc_hamming.pdf'),
                            transparent=True)
                
                
                # fig, ax = plt.subplots()
                # ax.set_xlabel('')
                # ax.set_ylabel('')
                # ax.set_xlim([size_min[structure], size_max[structure]])
                # ax.set_ylim(0,1)
                # cmiic_bernstein.plotScore('dagF', fig, ax, **plot_style_bernstein)
                # cmiic_gaussian.plotScore('dagF', fig, ax, **plot_style_gaussian)
                # cpc.plotScore('dagF', fig, ax, **plot_style_cpc)
                # elidan.plotScore('dag', fig, ax, **plot_style_cpc)
                # plt.savefig(os.path.join(apath, 'bernstein_gaussian_cpc_elidan_dagF.pdf'),
                            # transparent=True)
                
        elif distribution == 'dirichlet':
            bpath = os.path.join('../figures/', distribution, structure)
            Path(bpath).mkdir(parents=True, exist_ok=True)
            # c3off2_bernstein.setDataDistribution(distribution)
            cmiic_gaussian.setDataDistribution(distribution)
            cmiic_bernstein.setDataDistribution(distribution)
            cpc.setDataDistribution(distribution)
            
            # c3off2_bernstein.generate_data()
            cmiic_gaussian.generate_data()
            cmiic_bernstein.generate_data()
            cpc.generate_data()
            
            # print('c3off2 bernstein')
            # c3off2_bernstein.computeStructuralScore('skelF')
            # c3off2_bernstein.computeStructuralScore('hamming')
            # c3off2_bernstein.computeStructuralScore('dagF')
            
            print('cmiic gaussian')
            cmiic_gaussian.computeStructuralScore('skelF')
            cmiic_gaussian.computeStructuralScore('hamming')
            # cmiic_gaussian.computeStructuralScore('dagF')
            
            print('cmiic bernstein')
            cmiic_bernstein.computeStructuralScore('skelF')
            cmiic_bernstein.computeStructuralScore('hamming')
            # cmiic_bernstein.computeStructuralScore('dagF')
            
            print('cpc')
            cpc.computeStructuralScore('skelF')
            cpc.computeStructuralScore('hamming')
            # cpc.computeStructuralScore('dagF')
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], size_max[structure]])
            ax.set_ylim(0,1)
            # c3off2_bernstein.plotScore('skelF', fig, ax, **plot_style_3off2)
            cmiic_bernstein.plotScore('skelF', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotScore('skelF', fig, ax, **plot_style_gaussian)
            cpc.plotScore('skelF', fig, ax, **plot_style_cpc)
            # elidan.plotScore('skeleton', fig, ax, **plot_style_cpc)
            plt.savefig(os.path.join(bpath,'bernstein_gaussian_cpc_skelF.pdf'),
                        transparent=True)
            
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], size_max[structure]])
            ax.set_ylim(0,50)
            # c3off2_bernstein.plotScore('hamming', fig, ax, **plot_style_3off2)
            cmiic_bernstein.plotScore('hamming', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotScore('hamming', fig, ax, **plot_style_gaussian)
            cpc.plotScore('hamming', fig, ax, **plot_style_cpc)
            # elidan.plotScore('hamming', fig, ax, **plot_style_cpc)
            plt.savefig(os.path.join(bpath,'bernstein_gaussian_cpc_hamming.pdf'),
                        transparent=True)
            
            
            # fig, ax = plt.subplots()
            # ax.set_xlabel('')
            # ax.set_ylabel('')
            # ax.set_xlim([size_min[structure], size_max[structure]])
            # ax.set_ylim(0,1)
            # cmiic_bernstein.plotScore('dagF', fig, ax, **plot_style_bernstein)
            # cmiic_gaussian.plotScore('dagF', fig, ax, **plot_style_gaussian)
            # cpc.plotScore('dagF', fig, ax, **plot_style_cpc)
            # # elidan.plotScore('dag', fig, ax, **plot_style_cpc)
            # plt.savefig(os.path.join(bpath,'bernstein_gaussian_cpc_dagF.pdf'),
            #             transparent=True)
            
            
# elidan = Pipeline('elidan', max_parents=4, hc_restart=2)
# elidan.setDataDistribution(distribution, r=0.8)
# elidan.setDataStructure(structure)
# elidan.setNRestart(n_restart)
# elidan.setResultDomain(size_min, size_max, n_points)
# elidan.generate_data()

# elidan.computeStructuralScore('skeleton')
# elidan.computeStructuralScore('hamming')
# elidan.computeStructuralScore('dag')

# plot_style_elidan = {'linewidth':1.25, 'linestyle':'--', 'color':'blue', 'label':'elidan'}

