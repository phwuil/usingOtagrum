# doc
# http://openturns.github.io/openturns/master/index.html
# 
import matplotlib.pyplot as plt
import numpy as np
#from scipy.stats import random_correlation

import pyAgrum as gum
import openturns as ot
#import otagrum as otagr


def local_log_likelihood(D, gaussian_copula):
    log_numerator = gaussian_copula.computeLogPDF(D)
    
    parents_indices = [i for i in range(1, D.getDimension())]
    margin = gaussian_copula.getMarginal(parents_indices)
    log_denominator = margin.computeLogPDF(ot.Sample(np.array(D)[:, parents_indices]))
    
    #print("logden: ", log_denominator)
    #print("lognum: ", log_numerator)
    #print("lll: ", np.sum(log_numerator, axis=0) - np.sum(log_denominator, axis=0))
    
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
    M = D.getSize()
    #print("likelihood: ", log_likelihood(D, gaussian_copula, G))
    #print("penalty: ", penalty(M,G))
    return log_likelihood(D, gaussian_copula, G) - penalty(M, G)

def one_hill_climbing(D, gaussian_copula, G):
    best_score = bic_score(D, gaussian_copula, G)
    best_graph = G
    
    converged = False
    
    while not converged:
        converged = True
        #print("score : ", best_score)
        
        neighbor = find_neighbor(G)
        for n in neighbor:
            score = bic_score(D, gaussian_copula, n)
            print("score: ", score)
            if score > best_score:
                best_score = score
                best_graph = n
                converged = False
    return best_graph, best_score

def find_neighbor(G):
    for i in G.nodes():
        for j in G.nodes():
            if i!=j:
                newdag=gum.DAG(G)
                if G.existsArc(i,j):
                    newdag.eraseArc(i,j)
                    yield newdag
                    try:
                        newdag.addArc(j,i)
                    except gum.InvalidDirectedCycle:
                        continue
                    yield newdag
                else:
                    try:
                        newdag.addArc(i,j)
                    except gum.InvalidDirectedCycle:
                        continue
                    yield newdag

def create_empty_dag(N):
    dag = gum.DAG()
    for i in range(N):
        dag.addNode()
    return dag

def create_random_dag():
    return

def hill_climbing(D, restart=1):
    N = D.getDimension()
    
    kendall_tau = D.computeKendallTau()
    pearson_r = ot.CorrelationMatrix(np.sin((np.pi/2)*kendall_tau))
    
    gaussian_copula = ot.NormalCopula(pearson_r)
    
    # Empty graph
    G = create_empty_dag(N)
    score = 0
        
    best_graph = G
    best_score = score
    
    for r in range(restart):
        if r != 0:
            G = create_empty_dag()
        G, score = one_hill_climbing(D, gaussian_copula, G)
        if score > best_score:
            best_graph = G
            best_score = score
            
    return

ot.RandomGenerator.SetSeed(42)
np.random.seed(42)

M = 1000 # Size of the dataset
N = 4  # Dimension of the random vector

# Correlation matrix definition
R = ot.CorrelationMatrix(N)
R[0,1] = 0.5
R[0,2] = 0
R[0,3] = 0
R[1,2] = -0.5
R[1,3] = 0.5
R[2,3] = 0

print(R.isPositiveDefinite())
#print(R)
        
# Sampling from standard normal
D = ot.Normal([0] * N, [1] * N, R).getSample(M)
D_r = D.rank()/(D.getSize()+1)
print("D : ", D)
print("D_r : ", D_r)

hill_climbing(D_r)

# Scatter plot for each pair of variables
#plt.figure(figsize=(16,12))
#
#plt.subplot(6,2,1)
#plt.scatter(D[:,0], D[:,1], s=3)
#
#plt.subplot(6,2,2)
#plt.scatter(D_r[:,0], D_r[:,1], s=3)
#
#plt.subplot(6,2,3)
#plt.scatter(D[:,0], D[:,2], s=3)
#
#plt.subplot(6,2,4)
#plt.scatter(D_r[:,0], D_r[:,2], s=3)
#
#plt.subplot(6,2,5)
#plt.scatter(D[:,0], D[:,3], s=3)
#
#plt.subplot(6,2,6)
#plt.scatter(D_r[:,0], D_r[:,3], s=3)
#
#plt.subplot(6,2,7)
#plt.scatter(D[:,1], D[:,2], s=3)
#
#plt.subplot(6,2,8)
#plt.scatter(D_r[:,1], D_r[:,2], s=3)
#
#plt.subplot(6,2,9)
#plt.scatter(D[:,1], D[:,3], s=3)
#
#plt.subplot(6,2,10)
#plt.scatter(D_r[:,1], D_r[:,3], s=3)
#
#plt.subplot(6,2,11)
#plt.scatter(D[:,2], D[:,3], s=3)
#
#plt.subplot(6,2,12)
#plt.scatter(D_r[:,2], D_r[:,3], s=3)
#
#plt.savefig("scatter_plot_pair.pdf", transparent=True)
#plt.show()


#cor = ot.CorrelationMatrix(2)
#cor[0,1] = 0.5

#c = ot.NormalCopula(cor)

# Figure de la densité de la copula gaussienne
#fig = plt.figure(figsize=(6,6))
#ax = fig.add_subplot(111, projection='3d')
#
#u1 = u2 = np.linspace(start=0, stop=1, num=101)
#
#u1 = u1.reshape((u1.size,1))
#u2 = u2.reshape((u2.size,1))
#U1, U2 = np.meshgrid(u1, u2)
#
#flatU1 = np.reshape(U1, (U1.size,1))
#flatU2 = np.reshape(U2, (U2.size,1))
#flatU  = np.concatenate((flatU1, flatU2), axis=1)
#
#z = c.computePDF(flatU)
#Z = np.array(z).reshape(U1.shape)
#
#ax.plot_surface(U1, U2, Z, color="grey")