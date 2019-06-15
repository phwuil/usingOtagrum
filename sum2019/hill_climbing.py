#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import openturns as ot
import pyAgrum as gum
import score as sc
import dag_utils as du

def one_hill_climbing(D, gaussian_copula, G):
    best_graph = G
    best_score = sc.bic_score(D, gaussian_copula, G)

    tabu_list = []
    tabu_list.append(best_graph)
    
    converged = False
    
    while not converged:
        converged = True
        for n in du.find_neighbor(best_graph):
            if n not in tabu_list:
                score = sc.bic_score(D, gaussian_copula, n)
                # print("graph: ", n)
                # print("score: ", score)
                if score > best_score:
                    best_score = score
                    best_graph = gum.DAG(n)
                    converged = False
        tabu_list.append(best_graph)
    return best_graph, best_score



def hill_climbing(D, restart=1):
    N = D.getDimension()

    # Compute the estimate of the gaussian copula    
    kendall_tau = D.computeKendallTau()
    pearson_r = ot.CorrelationMatrix(np.sin((np.pi/2)*kendall_tau))
    
    # Create the gaussian copula with parameters pearson_r
    # if pearson_r isn't PSD, a regularization is done
    eps = 1e-6
    done = False
    while not done:
        try:    
            gaussian_copula = ot.NormalCopula(pearson_r)
            done = True
        except:
            print("Regularization")
            for i in range(pearson_r.getDimension()):
                for j in range(i):
                    pearson_r[i,j] /= 1 + eps
                    
    # Initialization
    G = du.create_empty_dag(N)
    score = sc.bic_score(D, gaussian_copula, G)
        
    best_graph = G
    best_score = score
    
    for r in range(restart):
        if r != 0:
            G = du.create_random_dag(N)
        G, score = one_hill_climbing(D, gaussian_copula, G)
        if score > best_score:
            best_graph = G
            best_score = score
            
    return best_graph, best_score