#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import openturns as ot
import numpy as np

def local_log_likelihood(D, gaussian_copula):
    parents_indices = [i for i in range(1, D.getDimension())]
    margin = gaussian_copula.getMarginal(parents_indices)
    
    log_numerator = gaussian_copula.computeLogPDF(D)
    log_denominator = margin.computeLogPDF(ot.Sample(np.array(D)[:, parents_indices]))
    
    return np.sum(log_numerator, axis=0) - np.sum(log_denominator, axis=0)


def log_likelihood(D, gaussian_copula, G):
    ll = 0
    for i in G.nodes():
        if G.parents(i) != set():
            indices = list(G.parents(i))
            indices = [i] + indices
            ll += local_log_likelihood(ot.Sample(np.array(D)[:, indices]),
                                       gaussian_copula.getMarginal(indices))
    return ll


def penalty(M, G):
    return (np.log(M)/2) * G.sizeArcs()


def bic_score(D, gaussian_copula, G):
    return log_likelihood(D, gaussian_copula, G) - penalty(D.getSize(), G)