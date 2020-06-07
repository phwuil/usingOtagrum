#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cmiic.continuousMIIC as cmiic

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

cmode = cmiic.CModeType.Gaussian
size_min = 50
size_max = 1000
n_points = 20
n_restart = 4
structure = 'alarm'
distribution = 'gaussian'
correlation = 0.8


#Asia
# alpha_domain = list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False),3))
# alpha_domain += list(np.round(np.linspace(0.01, 0.02, 5, endpoint=False), 3))
# alpha_domain += list(np.round(np.linspace(0.02, 0.06, 5), 3))
# alpha_domain = np.array(alpha_domain)

# beta_domain = list(np.round(np.linspace(-0.09, -0.01, 9), 3))
# beta_domain += list(np.round(np.linspace(-0.009, -0.001, 9),3))
# beta_domain += [0.]
# beta_domain += list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False),3))
# # beta_domain += list(np.round(np.linspace(0.01, 0.05, 5), 3))
# beta_domain = [0.0]
# beta_domain = np.array(beta_domain)

#Alarm
alpha_domain = list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False),3))
alpha_domain += list(np.round(np.linspace(0.01, 0.02, 5, endpoint=False), 3))
alpha_domain += list(np.round(np.linspace(0.02, 0.06, 5), 3))
alpha_domain = np.array(alpha_domain)

# beta_domain = list(np.round(np.linspace(-0.0001, 0.0, 2), 4))
# beta_domain += list(np.round(np.linspace(0.001, 0.01, 9, endpoint=False),3))
# beta_domain += list(np.round(np.linspace(0.01, 0.05, 5), 3))
beta_domain = [0.0]
beta_domain = np.array(beta_domain)

figSkel, axSkel = plt.subplots()
axSkelAB = axSkel.twinx()
axSkel.set_xlabel('')
axSkel.set_ylabel('')
axSkel.set_xlim([size_min, size_max])
axSkel.set_ylim(0,1)

figHam, axHam = plt.subplots()
axHamAB = axHam.twinx()
axHam.set_xlabel('')
axHam.set_ylabel('')
axHam.set_xlim([size_min, size_max])
axHam.set_ylim(0,50)

figAB, axAB = plt.subplots()
axAB.set_xlabel('')
axAB.set_ylabel('')
axAB.set_xlim([size_min, size_max])
# axAB.set_ylim(-0.09,0.06)
# axAB.set_yscale('log')

skelF_alpha = []
hamming_alpha = []

skelF_alpha_std = []
hamming_alpha_std = []
for alpha in alpha_domain:
    skelF_beta = []
    hamming_beta = []
    
    skelF_beta_std = []
    hamming_beta_std = []
    for beta in beta_domain:
        # pathToResults = '../results/alphaBeta/' #Asia
        pathToResults = '/home/lasserre/Desktop/res/alphaBeta/' # Alarm
        pathToResults = os.path.join(pathToResults, str(alpha), str(beta))
        pathToResults = os.path.join(pathToResults, distribution, structure)
        pathToResults = os.path.join(pathToResults, 'r'+str(correlation).replace('.', ''))
        pathToResults = os.path.join(pathToResults, 'scores')
        
        file_suffix = '_'.join(['cmiic', str(cmode), 'naive'])
        file_suffix += '_f' + str(size_min) + 't' + str(size_max) + \
                       'np' + str(n_points) + 'r' + str(n_restart) + '.csv'
                       
        skelF_data = np.loadtxt(os.path.join(pathToResults,'skelF_' + file_suffix),
                                   delimiter=',').transpose()
        hamming_data = np.loadtxt(os.path.join(pathToResults,'hamming_' + file_suffix),
                                     delimiter=',').transpose()     
        
        size = skelF_data[0]
        
        skelF = skelF_data[1]
        hamming = hamming_data[1]

        skelF_std = skelF_data[2]
        hamming_std = hamming_data[2]

        skelF_beta.append(skelF)
        hamming_beta.append(hamming)
        
        skelF_beta_std.append(skelF_std)
        hamming_beta_std.append(hamming_std)
        
        # axHam.plot(size, hamming)
        
    skelF_beta = np.array(skelF_beta)
    hamming_beta = np.array(hamming_beta)
    
    skelF_beta_std = np.array(skelF_beta_std)
    hamming_beta_std = np.array(hamming_beta_std)

    max_skelF_beta = np.max(skelF_beta, axis=0)
    # print(skelF_beta)
    # Find the argmax and if there are several, we take the one
    # with minimum std.
    argmax_beta = []
    min_skelF_beta_std = []
    for i,c in enumerate(skelF_beta.T):
        temp = np.argwhere(c == max_skelF_beta[i]).flatten()
        argmax_beta.append(temp)
    argmax_beta = np.array(argmax_beta)
    temp = []
    for i,am in enumerate(argmax_beta):
        if len(am) == 1:
            temp.append(beta_domain[am[0]])
        elif len(am) > 1:
            std = skelF_beta_std.T[i][am]
            betas = beta_domain[am]
            argmin_std = np.argwhere(std == np.min(std)).flatten()
            beta = betas[np.argmin(np.abs(betas[argmin_std]))]
            temp.append(beta)
        min_skelF_beta_std.append(np.min(skelF_beta_std.T[i][am]))
    del argmax_beta
    argmax_beta = temp[:]
    
    min_hamming_beta = np.min(hamming_beta, axis=0)
    # Find the argmin and if there are several, we take the one
    # with minimum std.
    argmin_beta = []
    min_hamming_beta_std = []
    for i,c in enumerate(hamming_beta.T):
        temp = np.argwhere(c == min_hamming_beta[i]).flatten()
        argmin_beta.append(temp)
    argmin_beta = np.array(argmin_beta)
    temp = []
    for i,am in enumerate(argmin_beta):
        if len(am) == 1:
            temp.append(beta_domain[am[0]])
        elif len(am) > 1:
            std = hamming_beta_std.T[i][am]
            betas = beta_domain[am]
            argmin_std = np.argwhere(std == np.min(std)).flatten()
            beta = betas[np.argmin(np.abs(betas[argmin_std]))]
            temp.append(beta)
        min_hamming_beta_std.append(np.min(hamming_beta_std.T[i][am]))
    del argmin_beta
    argmin_beta = temp[:]
        
    
    skelF_alpha.append(max_skelF_beta)
    hamming_alpha.append(min_hamming_beta)
    
    skelF_alpha_std.append(min_skelF_beta_std)
    hamming_alpha_std.append(min_hamming_beta_std)
    # axSkel.plot(size, max_skelF_beta)
    # axHam.plot(size, min_hamming_beta, color='black')

skelF_alpha = np.array(skelF_alpha)
hamming_alpha = np.array(hamming_alpha)

skelF_alpha_std = np.array(skelF_alpha_std)
hamming_alpha_std = np.array(hamming_alpha_std)

max_skelF_alpha = np.max(skelF_alpha, axis=0)
# argmax_skelF_alpha = np.argmax(skelF_alpha, axis=0)

argmax_skelF_alpha = []
for i,c in enumerate(skelF_alpha.T):
    temp = np.argwhere(c == max_skelF_alpha[i]).flatten()
    argmax_skelF_alpha.append(temp)
argmax_skelF_alpha = np.array(argmax_skelF_alpha)
temp = []
for i,am in enumerate(argmax_skelF_alpha):
    if len(am) == 1:
        temp.append(am[0])
    elif len(am) > 1:
        temp.append(am[np.argmin(skelF_alpha_std.T[i][am])])
    del argmax_skelF_alpha
    argmax_skelF_alpha = temp[:]

argmax_alpha = [alpha_domain[i] for i in argmax_skelF_alpha]
# argmax_beta = [beta_domain[i] for i in argmax_skelF_beta]

min_hamming_alpha = np.min(hamming_alpha, axis=0)
# argmin_hamming_alpha = np.argmin(hamming_alpha, axis=0)

argmin_hamming_alpha = []
for i,c in enumerate(hamming_alpha.T):
    temp = np.argwhere(c == min_hamming_alpha[i]).flatten()
    argmin_hamming_alpha.append(temp)
argmin_hamming_alpha = np.array(argmin_hamming_alpha)
temp = []
for i,am in enumerate(argmin_hamming_alpha):
    if len(am) == 1:
        temp.append(am[0])
    elif len(am) > 1:
        temp.append(am[np.argmin(hamming_alpha_std.T[i][am])])
    del argmin_hamming_alpha
    argmin_hamming_alpha = temp[:]
        
argmin_alpha = [alpha_domain[i] for i in argmin_hamming_alpha]
# argmin_beta = [beta_domain[i] for i in argmin_hamming_beta]

# Plot skelF max envelope with corresponding parameters
axSkel.plot(size, max_skelF_alpha, label=r'F-score', color='blue', linestyle='--')
axSkelAB.plot(size, argmax_alpha, label=r'$\alpha$', color='red', linestyle='-.')
axSkelAB.plot(size, argmax_beta, label=r'$\beta$', color='green', linestyle='-')
lines, labels = axSkel.get_legend_handles_labels()
lines2, labels2 = axSkelAB.get_legend_handles_labels()
axSkelAB.legend(lines + lines2, labels + labels2)
figSkel.savefig('skelf_cmiic_gaussian_alarm_gaussian_alpha.png', transparent=True)

# Plot Hamming min envelope with corresponding parameters
axHam.plot(size, min_hamming_alpha, label=r'Hamming', color='blue', linestyle='--')
axHamAB.plot(size, argmin_alpha, label=r'$\alpha$', color='red', linestyle='-.')
axHamAB.plot(size, argmin_beta, label=r'$\beta$', color='green', linestyle='-')
lines, labels = axHam.get_legend_handles_labels()
lines2, labels2 = axHamAB.get_legend_handles_labels()
axHamAB.legend(lines + lines2, labels + labels2)
figHam.savefig('hamming_cmiic_gaussian_alarm_gaussian_alpha.png', transparent=True)

# Plot parameters curves
axAB.plot(size, argmax_alpha, label=r'F-score $\alpha$')
axAB.plot(size, argmax_beta, label=r'F-score $\beta$')
axAB.plot(size, argmin_alpha, label=r'Hamming $\alpha$')
axAB.plot(size, argmin_beta, label=r'Hamming $\beta$')
axAB.legend()
figAB.savefig('alpha_cmiic_gaussian_alarm_gaussian_alpha.png', transparent=True)