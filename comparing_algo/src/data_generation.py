# -*- coding: utf-8 -*-

import openturns as ot
import otagrum as otagr

def generate_gaussian_data(ndag, size, r=0.8):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for k in range(order.getSize()):
        d = 1 + ndag.getParents(k).getSize()
        R = ot.CorrelationMatrix(d)
        for i in range(d):
            for j in range(i):
                R[i, j] = r
        jointDistributions.append(ot.NormalCopula(R))
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(size)
    return sample

def generate_gaussian_copulas(ndag, r=0.8):
    lcc = []
    for k in range(ndag.getSize()):
        d = 1 + ndag.getParents(k).getSize()
        R = ot.CorrelationMatrix(d)
        for i in range(d):                                                            
            for j in range(i):                                                        
                R[i, j] = r                                                           
        lcc.append(ot.Normal([0.0]*d, [1.0]*d, R).getCopula()) 
    return lcc

def generate_student_data(ndag, size, r=0.8):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for k in range(order.getSize()):
        d = 1 + ndag.getParents(k).getSize()
        R = ot.CorrelationMatrix(d)
        for i in range(d):
            for j in range(i):
                R[i, j] = r
        jointDistributions.append(ot.Student(5.0, [0.0]*d, [1.0]*d, R).getCopula())
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(size)
    return sample

def generate_dirichlet_data(ndag, size):
    order=ndag.getTopologicalOrder()
    jointDistributions=[]
    for k in range(order.getSize()):
        d = 1 + ndag.getParents(k).getSize()
        jointDistributions.append(ot.Dirichlet([(1.0+k)/(d+1) for k in range(d+1)]).getCopula())
    copula = otagr.ContinuousBayesianNetwork(ndag, jointDistributions)
    sample = copula.getSample(size)
    return sample

def generate_data(ndag, size, distribution, **kwargs):
    if distribution == "gaussian":
        sample = generate_gaussian_data(ndag, size, kwargs['r'])
    elif distribution == "student":
        sample = generate_student_data(ndag, size, kwargs['r'])
    elif distribution == "dirichlet":
        sample = generate_dirichlet_data(ndag, size)
    else:
        print("Wrong entry for the distribution !")
    return sample