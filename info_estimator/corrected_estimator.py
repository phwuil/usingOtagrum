import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path

import numpy as np
import time
from scipy.special import binom, beta
from scipy.stats import norm
import otagrum
import openturns as ot
from pathlib import Path

# Function definitions
def compute_true_information(rho):
    return - 0.5 * np.log(1 - rho**2)

def gamma(k, v1, v2):
    return binom(k-1, v1) * binom(k-1, v2) * beta(v1 + v2 + 1, 2*k - 1 - v1 - v2)

def compute_ttest(delta, n, d):
    k = otagrum.ContinuousTTest_GetK(n, d)
    A = np.sqrt(2) * n * delta
    B = (k * np.pi)**(d/2) * 2**(-(d+0.5))
    C = np.sqrt((k**2 * gamma2_sum(k))**d - 1)
    return (A - B) / C

def gamma2_sum(k):
    gamma2_sum = 0
    for v1 in range(k):
        for v2 in range(k):
            gamma2_sum += gamma(k, v1, v2)**2
    return gamma2_sum

def compute_p_value(t):
    return 2 * norm.cdf(-np.abs(t))

def is_indep(pvalue, alpha):
    return pvalue >= alpha

def save_dict(dictionary, location='.'):
    path = Path(location)
    path.mkdir(parents=True, exist_ok=True)
    for key in dictionary:
        name = str(key).zfill(7)
        np.savetxt(path/(name+'.csv'), dictionary[key], delimiter=',')

def load_dict(path, sizes):
    path = Path(path)
    dictionary = {}
    files_to_load = sorted([f for f in path.iterdir() if int(f.stem) in sizes],
                           key=lambda f: int(f.stem))
    for f in files_to_load:
        dictionary[int(f.stem)] = np.loadtxt(f, delimiter=',')
        print("\tLoaded file {}".format(f))
    return dictionary
        
def create_normal_distribution(correlation):
    # Generating 2D gaussian data with correlation of 0.8
    cm = ot.CorrelationMatrix([[          1, correlation],
                               [correlation,           1]])
    distribution = ot.Normal([0, 0], [1, 1], cm)
    return distribution

def create_uniform_distribution():
    distribution = ot.ComposedDistribution([ot.Uniform(0., 1.)]*2)
    return distribution

def compute_results(sizes, restarts, distribution, verbose=False):
    info_results = {}
    ttest_results = {}
    for size in sizes:
        start = time.time()
        size = int(size)
        if verbose:
            print("\tSize: {}".format(size))
        infos = []
        ttests = []
        for _ in range(restarts):
            sample = distribution.getSample(size)

            # Computing mutual information using otagrum
            icomputer = otagrum.CorrectedMutualInformation(sample)
            icomputer.setKMode(otagrum.CorrectedMutualInformation.KModeTypes_NoCorr)
            info = icomputer.compute2PtCorrectedInformation(0, 1)
            print("OTAGRUM: ", info)

            # Reproducing compute2PtCorrectedInformation without using otagrum
            K = otagrum.ContinuousTTest_GetK(sample.getSize(), sample.getDimension())
            bc = ot.EmpiricalBernsteinCopula(sample, K, False)
            I = bc.computeLogPDF(sample).computeMean()[0]
            # I = bc.computeLogPDF(bc.getCopulaSample()*(1.0 - SpecFunc.ScalarEpsilon)).computeMean()[0]
            # I = bc.computeLogPDF(sample*(1.0 - SpecFunc.ScalarEpsilon)).computeMean()[0]
            # I = -bc.computeEntropy()
            print("PYTHON: ", I)

            # Computing TTest with the obtained info estimator
            ttest = compute_ttest(info, size, 2)

            infos.append(info)
            ttests.append(ttest)

        info_results[size] = np.array(infos)
        ttest_results[size] = np.array(ttests)
        
        save_dict(info_results, Path('results')/mode/'info'/str(restarts))
        save_dict(ttest_results, Path('results')/mode/'ttest'/str(restarts))
        end = time.time()
        if verbose:
            print("\tElapsed time : {}".format(end - start))

    return info_results, ttest_results

def apply_to_dic(dictionary, method):
    new = {}
    for key in dictionary:
        new[key] = method(dictionary[key])
    return new


# Main
mode = 'uniform'
size_min, size_max = 100, 101
sizes = np.arange(size_min, size_max, 600)
print(sizes)
restarts = 5000

dir_prefix = Path('results')/mode
info_dir = dir_prefix/'info'/str(restarts)
ttest_dir = dir_prefix/'ttest'/str(restarts)

if mode == 'uniform':
    distribution = create_uniform_distribution()
elif mode == 'gaussian':
    distribution = create_normal_distribution(0.)

print("Checking existing results")
if not info_dir.exists() or not ttest_dir.exists():
    print("Found no results, computing them...")
    info_results, ttest_results = compute_results(sizes, restarts, distribution, verbose=True)
else:
    # Find which size files have been generated
    found_info_sizes = [int(f.stem) for f in info_dir.iterdir()]
    found_ttest_sizes = [int(f.stem) for f in ttest_dir.iterdir()]
    remaining_info_sizes = [size for size in sizes if size not in found_info_sizes]
    remaining_ttest_sizes = [size for size in sizes if size not in found_ttest_sizes]
    remaining_sizes = list(set(remaining_info_sizes + remaining_ttest_sizes))

    # Generate the remaining ones
    if remaining_sizes:
        print("Computing missing results...")
        remaining_sizes.sort()
        compute_results(remaining_sizes, restarts, distribution, verbose=True)

    # Load the results into dictionaries
    print("Loading data...")
    info_results = load_dict(info_dir, sizes)
    ttest_results = load_dict(ttest_dir, sizes)
    print("Done!")
    
# Compute mean and std
print("Computing mean and standard deviation...")
means = apply_to_dic(ttest_results, np.mean)
stds = apply_to_dic(ttest_results, np.std)
print("Done!")

fig, ax = plt.subplots()
# ax.errorbar(sizes, list(means.values()), list(stds.values()))
ax.plot(sizes, list(means.values()))
plt.savefig("mean_" + mode + ".pdf", transparent=True)
# plt.show()

fig, ax = plt.subplots()
ax.plot(sizes, list(stds.values()))
plt.savefig("std_" + mode + ".pdf", transparent=True)
# plt.show()

# Histogram animation
# fig, ax = plt.subplots()

# n, bins = np.histogram(ttest_results[size_min], bins=50)

# left = np.array(bins[:-1])
# right = np.array(bins[1:])
# bottom = np.zeros(len(left))
# top = bottom + n
# nrects = len(left)

# nverts = nrects*(1 + 3 + 1)
# verts = np.zeros((nverts, 2))
# codes = np.ones(nverts, int) * path.Path.LINETO
# codes[0::5] = path.Path.MOVETO
# codes[4::5] = path.Path.CLOSEPOLY
# verts[0::5, 0] = left
# verts[0::5, 1] = bottom
# verts[1::5, 0] = left
# verts[1::5, 1] = top
# verts[2::5, 0] = right
# verts[2::5, 1] = top
# verts[3::5, 0] = right
# verts[3::5, 1] = bottom

# barpath = path.Path(verts, codes)
# patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
# ax.add_patch(patch)

# ax.set_xlim(left[0], right[-1])
# ax.set_ylim(bottom.min(), top.max())

# patch = None

# def animate_histogram(i, dictionary):
    # keys = list(dictionary.keys())
    # n, bins = np.histogram(dictionary[keys[i]], 50)
    # top = bottom + n
    # verts[1::5, 1] = top
    # verts[2::5, 1] = top
    # return [patch, ]

# fig, ax = plt.subplots()
# barpath = path.Path(verts, codes)
# patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
# ax.add_patch(patch)

# ax.set_xlim(left[0], right[-1])
# ax.set_ylim(bottom.min(), top.max())
# ani = animation.FuncAnimation(fig, lambda i:animate_histogram(i, ttest_results),
                                    # len(sizes), repeat=False, blit=True)
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=5, bitrate=1800)
# ani.save('histograms.mp4', writer=writer)

# alpha = 0.1
    # bins = 40
    # X = np.linspace(-10, 10, 200)
    # plt.hist(currents, bins=bins, alpha=0.5, density=True)
    # plt.show()
    # plt.hist(ttests, bins=bins, alpha=0.5, density=True)
    # plt.plot(X, norm.pdf(X))
    # plt.show()

# if visualization:
    # plt.scatter(sample.getMarginal(0), sample.getMarginal(1), s=0.2)
    # plt.show()
# if do_print:
    # print("True value:  I(X;Y) = {}".format(true_I_XY))
    # print("Estimator: I(X;Y) = {}".format(current_I_XY))
    # print("Statistics: t-test = {}, p-value = {}, is independent ? {}".format(ttest,
                                                                              # p_value,
                                                                              # indep))
