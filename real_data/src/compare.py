import openturns as ot
import openturns.viewer as otv
import pyAgrum as gum
import otagrum as otagr

from time import time
from pathlib import Path
import os

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

import graph_utils as gu


#ot.ResourceMap.SetAsString("BernsteinCopulaFactory-SearchingMethod", "Approximate")
def pairs(data, filename):
    print("  Draw pairs")
    print("    Distribution")
    g = ot.Graph()
    pairs_data = ot.Pairs(data)
    pairs_data.setPointStyle('dot')
    g.add(pairs_data)
    view = otv.View(g,(800,800),square_axes=True)
    view.save(filename)
    print("Saving figure in {}".format(filename))
    view.close()
    print("    Copula")
    g = ot.Graph()
    pairs_data = ot.Pairs((data.rank() + 0.5) / data.getSize())
    pairs_data.setPointStyle('dot')
    g.add(pairs_data)
    view = otv.View(g,(800,800),square_axes=True)
    view.save(filename)
    print("Saving figure in {}".format(filename))
    view.close()

def plot_shd_diff(pdag_path, output_path):
    plt.clf()
    list_file = [e for e in pdag_path.iterdir() if e.is_file()]
    list_file.sort()

    shds = []
    for i in range(len(list_file)-1):
        mg1, names1 = gu.read_graph(list_file[i])
        mg2, names2 = gu.read_graph(list_file[i+1])

        sc = gum.StructuralComparator()
        sc.compare(mg1, names1, mg2, names2)

        shds.append(sc.shd())

    plt.plot(shds)
    plt.savefig(output_path.joinpath("shd_diff.pdf"), transparent=True)

def get_KS_marginals(data):
    print("Marginal KS")
    dimension = data.getDimension()
    KS = ot.KernelSmoothing(ot.Epanechnikov(), False, 0, False)
    marginals = [KS.build(data.getMarginal(i)) for i in range(dimension)]
    return marginals

def fullKS(data): 
    # First model: full KS
    print("Full KS")
    model = ot.KernelSmoothing(ot.Epanechnikov(), False, 0, False).build(data)
    return model


def IC_marginalKS(data, marginals):
    # Second model: marginal KS + independent copula
    print("Indep. copula")
    dimension = data.getDimension()
    model = ot.ComposedDistribution(marginals, ot.IndependentCopula(dimension))
    return model


def NC_marginalKS(data, marginals):
    # Third model: marginal KS + normal copula
    print("Normal copula")
    model = ot.ComposedDistribution(marginals, ot.NormalCopulaFactory().build(data))
    return model


def BC_marginalKS(data, marginals):
    # Fourth model: marginal KS + Bernstein copula
    print("Bernstein copula")
    # ot.Log.Show(ot.Log.INFO)
    model = ot.ComposedDistribution(marginals, ot.BernsteinCopulaFactory().build(data))
    return model


def CBN_PC(data, result_structure_path):
    print("CBN with PC")

    skeleton_path = result_structure_path.joinpath("skeleton")
    skeleton_path.mkdir(parents=True, exist_ok=True)

    pdag_path = result_structure_path.joinpath("pdag")
    pdag_path.mkdir(parents=True, exist_ok=True)

    dag_path = result_structure_path.joinpath("dag")
    dag_path.mkdir(parents=True, exist_ok=True)

    skeleton_file_name = "skeleton_" + str(size).zfill(7) + ".dot"
    skeleton_done = skeleton_path.joinpath(skeleton_file_name).exists()

    pdag_file_name = "pdag_" + str(size).zfill(7) + ".dot"
    pdag_done = pdag_path.joinpath(pdag_file_name).exists()

    dag_file_name = "dag_" + str(size).zfill(7) + ".dot"
    dag_done = dag_path.joinpath(dag_file_name).exists()

    alpha = 0.01
    conditioningSet = 4

    learner = otagr.ContinuousPC(data, conditioningSet, alpha)
    learner.setVerbosity(True)

    if not skeleton_done:
        skel = learner.learnSkeleton()
        gu.write_graph(skel, skeleton_path.joinpath("skeleton_" + str(size).zfill(7) + ".dot"));

    if not pdag_done:
        pdag = learner.learnPDAG()
        gu.write_graph(pdag, pdag_path.joinpath("pdag_" + str(size).zfill(7) + ".dot"));

    if not dag_done:
        dag = learner.learnDAG()
        gu.write_graph(dag, dag_path.joinpath("dag_" + str(size).zfill(7) + ".dot"));
    else:
        dag,names = gu.read_graph(dag_path.joinpath("dag_" + str(size).zfill(7) + ".dot"));
        dag = otagr.NamedDAG(dag, names)

    print("Learning parameters")
    factories = [ot.KernelSmoothing(ot.Epanechnikov()), ot.BernsteinCopulaFactory()]
    ot.Log.SetFile("log")
    ot.Log.Show(ot.Log.INFO)
    model = otagr.ContinuousBayesianNetworkFactory(factories,
                                                   dag,
                                                   alpha,
                                                   conditioningSet,
                                                   False).build(data)
    ot.Log.Show(ot.Log.INFO)
    return model

def visu(data, size_draw = 1000):
    pairs_path = Path("../figures/pairs")
    pairs_path.mkdir(parents=True, exist_ok=True)

    result_structure_path = Path("../results/structures/")
    result_structure_path.mkdir(parents=True, exist_ok=True)

    ot.Log.Show(ot.Log.NONE)

    t0 = time()
    t1 = time()

    size = data.getSize()
    dimension = data.getDimension()
    print("t=", t1 - t0, "s, speed=", size * dimension / (t1 - t0) * 1e-6, "Mfloat/s")

    size_draw = min(size, size_draw)
    data_draw = data[0:size_draw]

    print("Ref")
    filename = pairs_path.joinpath("pairs_ref_" + str(size).zfill(7) + ".pdf")
    pairs(data_draw, filename)

    KS_marginals = get_KS_marginals(data)

    # Model 1
    # model1 = fullKS(data)

    # filename = pairs_path.joinpath("pairs_fullKS_" + str(size).zfill(7) + ".pdf")
    # pairs(model1.getSample(size_draw), filename)

    # Model 2
    # model2 = IC_marginalKS(data, KS_marginals)

    # filename = pairs_path.joinpath("pairs_ICMKS_" + str(size).zfill(7) + ".pdf")
    # pairs(model2.getSample(size_draw), filename)

    # Model 3
    # model3 = NC_marginalKS(data, KS_marginals)

    # filename = pairs_path.joinpath("pairs_NCMKS_" + str(size).zfill(7) + ".pdf")
    # pairs(model3.getSample(size_draw), filename)

    # Model 4
    # model4 = BC_marginalKS(data, KS_marginals)

    # filename = pairs_path.joinpath("pairs_BCMKS" + str(size).zfill(7) + ".pdf")
    # pairs(model4.getSample(size_draw), filename)

    # Model 5
    model5 = CBN_PC(data, result_structure_path)
    print("sample")
    sample_draw = model5.getSample(size_draw)
    print(sample_draw)

    filename = pairs_path.joinpath("pairs_CBNPC" + str(size).zfill(7) + ".pdf")
    pairs(model5.getSample(size_draw), filename)

    ## print("Model 6")
    ## factories = [ot.KernelSmoothing(ot.Epanechnikov()), ot.NormalCopulaFactory(), ot.BernsteinCopulaFactory()]
    ## copula = otagr.ContinuousBayesianNetworkFactory(factories, dag, alpha, conditioningSet, True).build((data.rank()+1.0)/data.getSize())
    ## marginals = [ot.KernelSmoothing(ot.Epanechnikov(), False, 0, False).build(data.getMarginal(i)) for i in range(data.getDimension())]
    ## model6 = ot.ComposedDistribution(marginals, copula.getCopula())
    ## print("sample")
    ## sample_draw = model6.getSample(size_draw)
    ## print(sample_draw)
    ## pairs(sample_draw, "_model6_" + str(size).zfill(7))

def ttest_indep_threshold(alpha):
    return -norm.ppf(alpha/2)

if __name__ == "__main__":
    result_structure_path = Path("../results/structures/")
    result_structure_path.mkdir(parents=True, exist_ok=True)
    figure_structure_path = Path("../figures/structures/")
    figure_structure_path.mkdir(parents=True, exist_ok=True)

    result_score_path = Path("../results/scores/")
    result_score_path.mkdir(parents=True, exist_ok=True)
    figure_score_path = Path("../figures/scores/")
    figure_score_path.mkdir(parents=True, exist_ok=True)

    Path(os.path.join(result_score_path, "delta_1")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(figure_score_path, "delta_1")).mkdir(parents=True, exist_ok=True)

    file_name = "../data/Standard_coefficients_" + str(100000).zfill(7) + ".csv"

    data_ref = ot.Sample.ImportFromTextFile(file_name, ";")
    data_ref = data_ref.getMarginal(range(12))
    # data_ref = data_ref.getMarginal([2, 3, 4])
    data_ref = (data_ref.rank() + 0.5) / data_ref.getSize()

    # sizes = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]
    sizes = [10000]
    # sizes = np.arange(8000, 9000, 100, dtype=int)

    # ttest_values = {}
    # ttest_values['0 indep 1 | 11'] = []
    # ttest_values['8 indep 1'] = []
    # ttest_values['10 indep 11'] = []
    # ttest_values['2 indep 7 | 0'] = []
    # ttest_values['1 indep 7'] = []
    # ttest_values['0 indep 8 | 9, 4, 2'] = []

    # info_values = {}
    # info_values['0 indep 1 | 11'] = []
    # info_values['8 indep 1'] = []
    # info_values['10 indep 11'] = []
    # info_values['2 indep 7 | 0'] = []
    # info_values['1 indep 7'] = []
    # info_values['0 indep 8 | 9, 4, 2'] = []

    for size in sizes:
        print("#"*50)
        print("size=", size)

        data = ot.Sample(data_ref[0:size])
        visu(data)

        # ttest = otagr.ContinuousTTest(data)
        # ttest_values['0 indep 1 | 11'].append(ttest.getTTest(0, 1, [11]))
        # ttest_values['8 indep 1'].append(ttest.getTTest(8, 1, []))
        # ttest_values['10 indep 11'].append(ttest.getTTest(10, 11, []))
        # ttest_values['2 indep 7 | 0'].append(ttest.getTTest(2, 7, [0]))
        # ttest_values['1 indep 7'].append(ttest.getTTest(1, 7, []))
        # ttest_values['0 indep 8 | 9, 4, 2'].append(ttest.getTTest(0, 8, [9, 4, 2]))

        # info = otagr.CorrectedMutualInformation(data)
        # info_values['0 indep 1 | 11'].append(info.compute2PtCorrectedInformation(0, 1, [11]))
        # info_values['8 indep 1'].append(info.compute2PtCorrectedInformation(8, 1, []))
        # info_values['10 indep 11'].append(info.compute2PtCorrectedInformation(10, 11, []))
        # info_values['2 indep 7 | 0'].append(info.compute2PtCorrectedInformation(2, 7, [0]))
        # info_values['1 indep 7'].append(info.compute2PtCorrectedInformation(1, 7, []))
        # info_values['0 indep 8 | 9, 4, 2'].append(info.compute2PtCorrectedInformation(0, 8, [9, 4, 2]))

    # plt.clf()
    # for i,key in enumerate(ttest_values):
        # plt.plot(sizes, ttest_values[key], label=key)

    # plt.axhline(ttest_indep_threshold(0.1), color='black', ls='--', label='alpha=0.1')
    # plt.axhline(ttest_indep_threshold(0.05), color='black', ls=':', label='alpha=0.05')
    # plt.axhline(ttest_indep_threshold(0.01), color='black', label='alpha=0.01')

    # plt.legend()
    # plt.savefig("ttest_output.pdf", transparent=True)

    # plt.clf()
    # for i,key in enumerate(info_values):
        # plt.plot(sizes, info_values[key], label=key)

    # plt.legend()
    # plt.savefig("info_output.pdf", transparent=True)

    # plot_shd_diff(result_structure_path.joinpath("pdag"),
                  # figure_score_path.joinpath("delta_1"))
