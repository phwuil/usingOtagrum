# -*- coding: utf-8 -*-

import otagrum as otagr
from pipeline import Pipeline
import numpy as np
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

cmode = otagr.CorrectedMutualInformation.CModeTypes_Bernstein
size_min = 100
size_max = 1000
n_points = 20
n_restart = 3
structure = 'asia'
distribution = 'gaussian'
correlation = 0.8

alpha_domain = list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False), 3))
alpha_domain += list(np.round(np.linspace(0.01, 0.02, 5, endpoint=False), 3))
alpha_domain += list(np.round(np.linspace(0.02, 0.05, 4),3))

# beta_domain = list([0.0])
beta_domain = list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False), 3))
beta_domain += list(np.round(np.linspace(0.01, 0.05, 5), 3))

# alpha_domain = list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False),3))
# alpha_domain += list(np.round(np.linspace(0.01, 0.02, 5, endpoint=False), 3))
# alpha_domain += list(np.round(np.linspace(0.02, 0.06, 5), 3))

# beta_domain = list(np.round(np.linspace(-0.07, -0.05, 2, endpoint=False), 3))
# beta_domain += list(np.round(np.linspace(-0.05, -0.02, 15, endpoint=False), 3))
# beta_domain += list(np.round(np.linspace(-0.02, -0.01, 2), 3))
# beta_domain += list(np.round(np.linspace(-0.009, -0.001, 9),3))
# beta_domain += [0.0]
# beta_domain += list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False),3))
# beta_domain += list(np.round(np.linspace(0.01, 0.05, 5), 3))

for alpha in alpha_domain:
    print("Alpha : ", alpha)
    figSkel, axSkel = plt.subplots()
    axSkel.set_xlabel('')
    axSkel.set_ylabel('')
    # axSkel.set_xlim([size_min, size_max])
    axSkel.set_ylim(0,1)
    
    figHam, axHam = plt.subplots()
    axHam.set_xlabel('')
    axHam.set_ylabel('')
    # axHam.set_xlim([size_min, size_max])
    axHam.set_ylim(0,90)
    for beta in beta_domain:
        print("\tBeta : ", beta)
        
        cmiic_gaussian = Pipeline('cmiic', cmode=cmode, kmode=otagr.CorrectedMutualInformation.KModeTypes_Naive)
        cmiic_gaussian.setDataDistribution(distribution, r=correlation)
        cmiic_gaussian.setDataStructure(structure)
        cmiic_gaussian.setResultDomain(size_min, size_max, n_points, n_restart)
        cmiic_gaussian.generate_data()
        cmiic_gaussian.setKAlpha(alpha)
        cmiic_gaussian.setKBeta(beta)
        cmiic_gaussian.setResultDirPrefix(os.path.join("results/alphaBeta/",
                                                       str(alpha),
                                                       str(beta)))
        
        cmiic_gaussian.computeStructuralScore('skelF')
        cmiic_gaussian.computeStructuralScore('hamming')
    
        cmiic_gaussian.plotScore('skelF', figSkel, axSkel, label=str(beta))
        cmiic_gaussian.plotScore('hamming', figHam, axHam, label=str(beta))
    
    thePath = '../figures/alphaBeta/' + str(alpha) + '/'
    Path(thePath).mkdir(parents=True, exist_ok=True)
    figSkel.savefig(thePath + structure + '_' + distribution + '_gaussian_skelF.pdf',
                transparent=True)
    figHam.savefig(thePath + structure + '_' + distribution + '_gaussian_hamming.pdf',
                transparent=True)
