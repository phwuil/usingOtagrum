#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import openturns as ot
import numpy as np

def local_log_likelihood(D, gaussian_copula):
    if gaussian_copula.getDimension()==2:
        return gaussian_copula.computeLogPDF(D).computeMean()[0] * D.getSize()
    parents_indices = [i for i in range(1, D.getDimension())]
    margin = gaussian_copula.getMarginal(parents_indices)
    log_numerator = gaussian_copula.computeLogPDF(D).computeMean()[0]
    log_denominator = margin.computeLogPDF(D.getMarginal(parents_indices)).computeMean()[0]
    return D.getSize() * (log_numerator - log_denominator)


def log_likelihood(D, gaussian_copula, G):
    ll = 0
    for i in G.nodes():
        if G.parents(i) != set():
            indices = list(G.parents(i))
            indices = [i] + indices
            ll += local_log_likelihood(D.getMarginal(indices),
                                       gaussian_copula.getMarginal(indices))
    return ll


def penalty(M, G):
    return (np.log(M)/2) * G.sizeArcs()


def bic_score(D, gaussian_copula, G):
    return log_likelihood(D, gaussian_copula, G) - penalty(D.getSize(), G)
