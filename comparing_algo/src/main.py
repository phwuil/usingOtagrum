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
mpl.rcParams.update({'font.size': 20})

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
# distributions = ['student']
structures = ['asia', 'alarm']
# structures = ['alarm']

size_min = {'asia':100, 'alarm':100}
size_max = {'asia':10000, 'alarm':10000}
n_points = {'asia':10, 'alarm': 10}
n_restart = {'asia': 5, 'alarm':2}
ylim = {'asia': 10, 'alarm':50}

correlations = np.round(np.linspace(0.4, 0.8, 2), decimals=1)

cmiic_gaussian = Pipeline('cmiic', cmode=cmi.CModeTypes_Gaussian, kmode=cmi.KModeTypes_Naive)
cmiic_bernstein = Pipeline('cmiic', cmode=cmi.CModeTypes_Bernstein, kmode=cmi.KModeTypes_Naive)
cpc = Pipeline('cpc', binNumber=5, alpha=0.05)

plot_style_bernstein = {'linewidth':2., 'linestyle':'-.', 'color':'green', 'label':'bernstein'}
plot_style_gaussian = {'linewidth':2., 'linestyle':'--', 'color':'orange', 'label':'gaussian'}
plot_style_cpc = {'linewidth':2., 'linestyle':':', 'color':'red', 'label':'cpc'}

for structure in structures:
    print('Structure :', structure)
    cmiic_gaussian.setDataStructure(structure)
    cmiic_bernstein.setDataStructure(structure)
    cpc.setDataStructure(structure)
    
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
                cmiic_gaussian.setDataDistribution(distribution, r=correlation)
                cmiic_bernstein.setDataDistribution(distribution, r=correlation)
                cpc.setDataDistribution(distribution, r=correlation)
                
                cmiic_gaussian.generate_data()
                cmiic_bernstein.generate_data()
                cpc.generate_data()
                
                print('cmiic gaussian')
                cmiic_gaussian.computeStructuralScore('skelF')
                cmiic_gaussian.computeStructuralScore('hamming')
                
                print('cmiic bernstein')
                cmiic_bernstein.computeStructuralScore('skelF')
                cmiic_bernstein.computeStructuralScore('hamming')
                
                print('cpc')
                cpc.computeStructuralScore('skelF')
                cpc.computeStructuralScore('hamming')

                fig, ax = plt.subplots()
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim([size_min[structure], size_max[structure]])
                ax.set_ylim(0,1)
                cmiic_bernstein.plotScore('skelF', fig, ax, **plot_style_bernstein)
                cmiic_gaussian.plotScore('skelF', fig, ax, **plot_style_gaussian)
                cpc.plotScore('skelF', fig, ax, **plot_style_cpc)
                # elidan.plotScore('skeleton', fig, ax, **plot_style_cpc)
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
                # elidan.plotScore('hamming', fig, ax, **plot_style_cpc)
                plt.savefig(os.path.join(apath,
                                         '_'.join(['hamming', distribution , structure]) +'.pdf'),
                        transparent=True)
                
                
        elif distribution == 'dirichlet':
            bpath = os.path.join('../figures/', distribution, structure)
            Path(bpath).mkdir(parents=True, exist_ok=True)
            cmiic_gaussian.setDataDistribution(distribution)
            cmiic_bernstein.setDataDistribution(distribution)
            cpc.setDataDistribution(distribution)
            
            cmiic_gaussian.generate_data()
            cmiic_bernstein.generate_data()
            cpc.generate_data()
            
            print('cmiic gaussian')
            print("alpha : ", cmiic_gaussian.kalpha)
            cmiic_gaussian.computeStructuralScore('skelF')
            cmiic_gaussian.computeStructuralScore('hamming')
            
            print('cmiic bernstein')
            cmiic_bernstein.computeStructuralScore('skelF')
            cmiic_bernstein.computeStructuralScore('hamming')
            
            print('cpc')
            cpc.computeStructuralScore('skelF')
            cpc.computeStructuralScore('hamming')
            
            fig, ax = plt.subplots()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([size_min[structure], size_max[structure]])
            ax.set_ylim(0,1)
            cmiic_bernstein.plotScore('skelF', fig, ax, **plot_style_bernstein)
            cmiic_gaussian.plotScore('skelF', fig, ax, **plot_style_gaussian)
            cpc.plotScore('skelF', fig, ax, **plot_style_cpc)
            # elidan.plotScore('skeleton', fig, ax, **plot_style_cpc)
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
            # elidan.plotScore('hamming', fig, ax, **plot_style_cpc)
            plt.savefig(os.path.join(bpath,
                                     '_'.join(['hamming', distribution , structure]) +'.pdf'),
                        transparent=True)
            
            
            
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
