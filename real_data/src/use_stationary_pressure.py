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
    distribution = ot.ComposedDistribution(marginals, copula)
    print("t=", time() - t0, "s")
    return distribution
    # return copula

def KSG_learning(data):
    print("Build KSG coefficients distribution")
    size = data.getSize()
    dimension = data.getDimension()
    t0 = time()
    marginals = [ot.HistogramFactory().build(data.getMarginal(i)) for i in range(dimension)]
    plot_marginals("KSG_marginals", marginals)
    copula = ot.NormalCopulaFactory().build(data)
    distribution = ot.ComposedDistribution(marginals, copula)
    print("t=", time() - t0, "s")
    return distribution

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
    with open("dags/new_MIIC_dag_{}.dot".format(alpha), "w") as f:
        f.write(dag.toDot())
    print("    t=", time() - t1, "s")

    cbn = CBN_parameter_learning(data, dag)
    # plot_marginals("marginals_MIIC", marginals)
    print("t=", time() - t0, "s")
    # distribution = ot.ComposedDistribution(marginals, cbn)
    return cbn

def CBN_parameter_learning(data, dag):
    size = data.getSize()
    dimension = data.getDimension()
    print("    Learning parameters")
    t1 = time()
    print("        Learning the CBN parameters")
    t2 = time()
    ot.ResourceMap.SetAsUnsignedInteger("BernsteinCopulaFactory-kFraction", 2)
    ot.ResourceMap.SetAsUnsignedInteger("BernsteinCopulaFactory-MinM", size//2-2)
    ot.ResourceMap.SetAsUnsignedInteger("BernsteinCopulaFactory-MaxM", size//2-1)
    cbn = otagrum.ContinuousBayesianNetworkFactory(ot.HistogramFactory(),
                                                   ot.BernsteinCopulaFactory(),
                                                   dag, 0, 0, False).build(data)
    print("        t=", time() - t2, "s")
    print("        Learning the marginal parameters")
    t2 = time()
    # marginals = [ot.KernelSmoothing().build(data.getMarginal(i)) for i in range(dimension)]
    # marginals = [ot.HistogramFactory().build(data.getMarginal(i)) for i in range(dimension)]
    print("        t=", time() - t2, "s")
    print("    t=", time() - t1, "s")
    return cbn

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
    with open("dags/new_dag_CPC_{}.dot".format(alpha), "w") as f:
        f.write(dag.toDot())
    print("    t=", time() - t1, "s")

    cbn = CBN_parameter_learning(data, dag)
    print("t=", time() - t0, "s")
    # distribution = ot.ComposedDistribution(marginals, cbn)
    return cbn

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
    with open("dags/new_dag_BIC_{}_{}_{}.dot".format(max_parents,
                                                 restart,
                                                 tabu_list_size), "w") as f:
        f.write(dag.toDot())
    print("    t=", time() - t1, "s")

    cbn = CBN_parameter_learning(data, dag)
    print("t=", time() - t0, "s")
    # distribution = ot.ComposedDistribution(marginals, cbn)
    return cbn

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

def generate_CBN_sample(name, cbn, size):
    # get the corresponding output distribution
    print("Generating data")
    t0 = time()
    coefficients = cbn.getSample(size)
    print("t=", time() - t0, "s")

    # print("Transforming data")
    # t0 = time()
    # coefficients = transform_data(coefficients, marginals)
    coefficients.exportToCSVFile("new_coefficients_{}.csv".format(name))
    # print("t=", time() - t0, "s")

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
    dist_stationary_pressure = ot.KernelSmoothing(ot.Normal(), False, 1000000, True).build(pressure)
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
dist_ref = ot.KernelSmoothing(ot.Normal(), False, 1000000, True).build(pressure_ref)
graph = dist_ref.drawPDF()
print("t=", time() - t0, "s")
ref_q1 = pressure_ref.computeQuantile(0.01)

ks_distribution = KS_learning(data)
ks_sample = generate_sample("KS", ks_distribution, sample_size)
ks_pressure = generate_pressure("KS", ks_sample)
draw_pressure(graph, ks_pressure)
ks_q1 = ks_pressure.computeQuantile(0.01)

ksb_distribution = KSB_learning(data)
ksb_sample = generate_sample("KSB", ksb_distribution, sample_size)
ksb_pressure = generate_pressure("KSB", ksb_sample)
draw_pressure(graph, ksb_pressure)
ksb_q1 = ksb_pressure.computeQuantile(0.01)

ksg_distribution = KSG_learning(data)
ksg_sample = generate_sample("KSG", ksg_distribution, sample_size)
ksg_pressure = generate_pressure("KSG", ksg_sample)
draw_pressure(graph, ksg_pressure)
ksg_q1 = ksg_pressure.computeQuantile(0.01)

for alpha in [0.01, 0.05, 0.1, 0.5]:
# for alpha in [0.05, 0.1, 0.5]:
    print("alpha = ", alpha)
    graph_cp = copy.copy(graph)
    miic_cbn = MIIC_learning(data, alpha)
    miic_sample = generate_CBN_sample("MIIC_{}".format(alpha), miic_cbn, sample_size)
    miic_pressure = generate_pressure("MIIC_{}".format(alpha), miic_sample)
    draw_pressure(graph_cp, miic_pressure)
    miic_q1 = miic_pressure.computeQuantile(0.01)
    graph_cp.setColors(["black", "red", "blue", "green", "orange"])
    graph_cp.setLegends(["Ref ({:.2e})".format(ref_q1[0]),
                         "KS ({:.2e})".format(ks_q1[0]),
                         "Bern ({:.2e})".format(ksb_q1[0]),
                         "Gauss ({:.2e})".format(ksg_q1[0]),
                         "MIIC_{} ({:.2e})".format(alpha,miic_q1[0])])
    # graph_cp.setLegendFontSize(0.1)
    view = otv.View(graph_cp)
    view.save("new_stationary_pressure_pdf_MIIC_{}.png".format(alpha))
    view.close()

for alpha in [0.01, 0.05, 0.1]:
    print("alpha = ", alpha)
    graph_cp = copy.copy(graph)
    cpc_cbn = CPC_learning(data, 5, alpha)
    cpc_sample = generate_CBN_sample("CPC_{}".format(alpha), cpc_cbn, sample_size)
    cpc_pressure = generate_pressure("CPC_{}".format(alpha), cpc_sample)
    draw_pressure(graph_cp, cpc_pressure)
    cpc_q1 = cpc_pressure.computeQuantile(0.01)
    graph_cp.setColors(["black", "red", "blue", "green", "orange"])
    graph_cp.setLegends(["Ref ({:.2e})".format(ref_q1[0]),
                         "KS ({:.2e})".format(ks_q1[0]),
                         "Bern ({:.2e})".format(ksb_q1[0]),
                         "Gauss (:.2e{})".format(ksg_q1[0]),
                         "CPC_{} (:.2e{})".format(alpha, cpc_q1[0])])
    view = otv.View(graph_cp)
    view.save("new_stationary_pressure_pdf_CPC_{}.png".format(alpha))
    view.close()

bic_cbn = BIC_learning(data)
bic_sample = generate_CBN_sample("BIC", bic_cbn, sample_size)
bic_pressure = generate_pressure("BIC", bic_sample)
draw_pressure(graph, bic_pressure)
bic_q1 = cpc_pressure.computeQuantile(0.01)
# view = otv.View(graph)
# view.save("new_stationary_pressure_pdf_BIC.png".format(alpha))
# view.close()

# Plot
graph.setColors(["black", "red", "blue", "green", "orange"])
graph.setLegends(["Ref ({:.2e})".format(ref_q1[0]),
                  "KS ({:.2e})".format(ks_q1[0]),
                  "Bern ({:.2e})".format(ksb_q1[0]),
                  "Gauss ({:.2e})".format(ksg_q1[0]),
                  "BIC ({:.2e})".format(bic_q1[0])])
view = otv.View(graph)
view.save("new_stationary_pressure_pdf_BIC.png")
view.close()
# ot.Show(graph_cp)
