import copy
import numpy as np
import openturns as ot
import openturns.viewer as otv
import otagrum
from time import time
from stationary_pressure import StationaryPressure

def KS_learning(data):
    # Naive estimation of the coefficients distribution using
    # a multivariate kernel smoothing
    print("Build KS coefficients distribution")
    t0 = time()
    distribution = ot.KernelSmoothing().build(data)
    print("t=", time() - t0, "s")
    return distribution

def KSB_learning(data):
    # Less naive estimation of the coefficients distribution using
    # univariate kernel smoothing for the marginals and a Bernstein copula
    print("Build KSB coefficients distribution")
    size = data.getSize()
    dimension = data.getDimension()
    t0 = time()
    marginals = [ot.HistogramFactory().build(data.getMarginal(i)) for i in range(dimension)]
    # marginals = [ot.KernelSmoothing().build(data.getMarginal(i)) for i in range(dimension)]
    plot_marginals("KSB_marginals", marginals)
    copula = ot.EmpiricalBernsteinCopula(data, size)
    #copula = ot.BernsteinCopulaFactory().build(data)
    # distribution = ot.ComposedDistribution(marginals, copula)
    print("t=", time() - t0, "s")
    # return distribution
    return copula

def plot_marginals(name, marginals):
    g = ot.Graph('', '', '', True)

    for m in marginals[:3]:
        g.add(m.drawPDF())

    g.setDefaultColors()
    # Saving figure
    view = otv.View(g)
    view.save(name)
    view.close()
    

def MIIC_learning(data, alpha):
    # Try an estimation of the coefficients distribution using
    # univariate kernel smoothing for the marginals and MIIC to learn the structure
    # of dependence parameterized by Bernstein copula
    dimension = data.getDimension()
    print("Build MIIC coefficients distribution")
    t0 = time()
    print("    Learning structure")
    t1 = time()
    learner = otagrum.ContinuousMIIC(data)
    learner.setAlpha(alpha)
    dag = learner.learnDAG()
    print("Nodes: ", dag.getDAG().sizeArcs())
    with open("dags/MIIC_dag_{}.dot".format(alpha), "w") as f:
        f.write(dag.toDot())
    print("    t=", time() - t1, "s")

    marginals, cbn = CBN_parameter_learning(data, dag)
    plot_marginals("marginals_MIIC", marginals)
    print("t=", time() - t0, "s")
    # distribution = ot.ComposedDistribution(marginals, cbn)
    return marginals, cbn

def CBN_parameter_learning(data, dag):
    dimension = data.getDimension()
    print("    Learning parameters")
    t1 = time()
    print("        Learning the CBN parameters")
    t2 = time()
    cbn = otagrum.ContinuousBayesianNetworkFactory([ot.BernsteinCopulaFactory()],
                                                   dag, 0, 0, True).build(data)
    print("        t=", time() - t2, "s")
    print("        Learning the marginal parameters")
    t2 = time()
    # marginals = [ot.KernelSmoothing().build(data.getMarginal(i)) for i in range(dimension)]
    marginals = [ot.HistogramFactory().build(data.getMarginal(i)) for i in range(dimension)]
    print("        t=", time() - t2, "s")
    print("    t=", time() - t1, "s")
    return marginals, cbn

def CPC_learning(data, maxCondSet=5, alpha=0.1):
    # Try an estimation of the coefficients distribution using
    # univariate kernel smoothing for the marginals and PC to learn the structure
    # of dependence parameterized by Bernstein copula
    dimension = data.getDimension()
    print("Build CPC coefficients distribution")
    t0 = time()
    print("    Learning structure")
    t1 = time()
    learner = otagrum.ContinuousPC(data, maxCondSet, alpha)
    dag = learner.learnDAG()
    with open("dags/dag_CPC_{}.dot".format(alpha), "w") as f:
        f.write(dag.toDot())
    print("    t=", time() - t1, "s")

    marginals, cbn = CBN_parameter_learning(data, dag)
    print("t=", time() - t0, "s")
    # distribution = ot.ComposedDistribution(marginals, cbn)
    return marginals, cbn

def BIC_learning(data, max_parents=3, restart=1, tabu_list_size=2):
    # Try an estimation of the coefficients distribution using
    # univariate kernel smoothing for the marginals and PC to learn the structure
    # of dependence parameterized by Bernstein copula
    dimension = data.getDimension()
    print("Build BIC coefficients distribution")
    t0 = time()
    print("    Learning structure")
    t1 = time()
    learner = otagrum.TabuList(data, max_parents, restart, tabu_list_size)
    dag = learner.learnDAG()
    with open("dags/dag_BIC_{}_{}_{}.dot".format(max_parents,
                                                 restart,
                                                 tabu_list_size), "w") as f:
        f.write(dag.toDot())
    print("    t=", time() - t1, "s")

    marginals, cbn = CBN_parameter_learning(data, dag)
    print("t=", time() - t0, "s")
    # distribution = ot.ComposedDistribution(marginals, cbn)
    return marginals, cbn

def transform_data(data, marginals):
    size = data.getSize()
    dimension = data.getDimension()
    t_data = []
    for p in data:
        t_p = []
        for i in range(dimension):
            t_p.append(marginals[i].computeQuantile(p[i]))
        t_data.append(t_p)
    t_data = ot.Sample(np.array(t_data).reshape(size, dimension))
    return t_data

def generate_CBN_sample(name, marginals, cbn, size):
    # get the corresponding output distribution
    print("Generating data")
    t0 = time()
    coefficients = cbn.getSample(size)
    print("t=", time() - t0, "s")

    print("Transforming data")
    t0 = time()
    # coefficients = transform_data(coefficients, marginals)
    coefficients.exportToCSVFile("coefficients_{}.csv".format(name))
    print("t=", time() - t0, "s")

    return coefficients

def generate_sample(name, distribution, size):
    # get the corresponding output distribution
    print("Generate data")
    t0 = time()
    sample = distribution.getSample(size)
    sample.exportToCSVFile(name)
    print("t=", time() - t0, "s")
    return sample

def generate_pressure(name, data):
    print("Generate pressure")
    t0 = time()
    pressure = stationary_pressure(data)
    pressure.exportToCSVFile(name)
    print("t=", time() - t0, "s")
    return pressure

def draw_pressure(graph, pressure):
    print("Build and draw pressure distribution")
    t0 = time()
    dist_stationary_pressure = ot.KernelSmoothing().build(pressure)
    print("t=", time() - t0, "s")
    graph.add(dist_stationary_pressure.drawPDF())

stationary_pressure = ot.Function(StationaryPressure())
ot.ResourceMap.SetAsUnsignedInteger("KernelSmoothing-BinNumber", 1000000000)

# Load the KL coefficients database
print("Load ref. coefficients")
t0 = time()
KL_coefficients_ref = ot.Sample.ImportFromTextFile("Standard_coefficients_1000000.csv", ";")
print("t=", time() - t0, "s")

# Parameters
learning_size = 10000
sample_size = 200

# Selecting data
data = KL_coefficients_ref[:learning_size]

# Get the corresponding output
pressure_ref = ot.Sample.ImportFromTextFile("pressure_ref.csv", ";")
t0 = time()
dist_ref = ot.KernelSmoothing().build(pressure_ref)
graph = dist_ref.drawPDF()
print("t=", time() - t0, "s")

ks_distribution = KS_learning(data)
ks_sample = generate_sample("KS", ks_distribution, sample_size)
ks_pressure = generate_pressure("KS", ks_sample)
draw_pressure(graph, ks_pressure)

ksb_distribution = KSB_learning(data)
ksb_sample = generate_sample("KSB", ksb_distribution, sample_size)
ksb_pressure = generate_pressure("KSB", ksb_sample)
draw_pressure(graph, ksb_pressure)

for alpha in [0.]:
    graph_cp = copy.copy(graph)
    miic_marginals, miic_cbn = MIIC_learning(data, alpha)
    miic_sample = generate_CBN_sample("MIIC_{}".format(alpha), miic_marginals, miic_cbn, sample_size)
    miic_pressure = generate_pressure("MIIC_{}".format(alpha), miic_sample)
    draw_pressure(graph_cp, miic_pressure)
    graph_cp.setColors(["black", "red", "blue", "green"])
    graph_cp.setLegends(["Ref", "KS", "Bern", "MIIC_{}".format(alpha)])
    view = otv.View(graph_cp)
    view.save("stationary_pressure_pdf_MIIC_{}.png".format(alpha))
    view.close()

# for alpha in [0.01, 0.05, 0.1]:
    # graph_cp = copy.copy(graph)
    # cpc_marginals, cpc_cbn = CPC_learning(data, 5, alpha)
    # cpc_sample = generate_CBN_sample("CPC", cpc_marginals, cpc_cbn, sample_size)
    # cpc_pressure = generate_pressure("CPC", cpc_sample)
    # draw_pressure(graph_cp, cpc_pressure)
    # graph_cp.setColors(["black", "red", "blue", "green"])
    # graph_cp.setLegends(["Ref", "KS", "Bern", "CPC{}".format(alpha)])
    # view = otv.View(graph_cp)
    # view.save("stationary_pressure_pdf_CPC_{}.png".format(alpha))
    # view.close()

# bic_marginals, bic_cbn = BIC_learning(data)
# bic_sample = generate_CBN_sample("BIC", bic_marginals, bic_cbn, sample_size)
# bic_pressure = generate_pressure("BIC", bic_sample)
# draw_pressure(graph, bic_pressure)

# print("Q1% ref =", pressure_ref.computeQuantile(0.01))
# print("Q1% KS  =", ks_pressure.computeQuantile(0.01))
# print("Q1% Bern=", ksb_pressure.computeQuantile(0.01))
# print("Q1% MIIC=", miic_pressure.computeQuantile(0.01))
# print("Q1% PC=", cpc_pressure.computeQuantile(0.01))
# print("Q1% BIC=", bic_pressure.computeQuantile(0.01))

# Plot
# graph.setColors(["black", "red", "blue", "green", "orange", "violet"])
# graph.setLegends(["Ref", "KS", "Bern", "MIIC", "PC", "BIC"])
# view = otv.View(graph_cp)
# view.save("stationary_pressure_pdf.png")
# view.close()
# ot.Show(graph_cp)
