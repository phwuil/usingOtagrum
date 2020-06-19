import openturns as ot
import openturns.viewer as otv
from time import time
import otagrum as otagr

#ot.ResourceMap.SetAsString("BernsteinCopulaFactory-SearchingMethod", "Approximate")
def pairs(data, filename):
    print("  Draw pairs")
    print("    Distribution")
    g = ot.Graph()
    pairs_data = ot.Pairs(data)
    pairs_data.setPointStyle('dot')
    g.add(pairs_data)
    view = otv.View(g,(800,800),square_axes=True)
    view.save("Distribution_pairs" + filename + ".png")
    view.close()
    print("    Copula")
    g = ot.Graph()
    pairs_data = ot.Pairs((data.rank() + 0.5) / data.getSize())
    pairs_data.setPointStyle('dot')
    g.add(pairs_data)
    view = otv.View(g,(800,800),square_axes=True)
    view.save("Copula_pairs" + filename + ".png")
    view.close()
    
def visu(N, data_ref, size_draw = 1000):
    print("Load data")
    data = ot.Sample(data_ref[0:N])
    ot.Log.Show(ot.Log.NONE)
    t0 = time()
    t1 = time()
    size = data.getSize()
    dimension = data.getDimension()
    print("t=", t1 - t0, "s, speed=", size * dimension / (t1 - t0) * 1e-6, "Mfloat/s")
    size_draw = min(size, size_draw)
    data_draw = data[0:size_draw]
    print("Ref")
    pairs(data_draw, "_ref_" + str(N).zfill(7))
    # First model: full KS
    print("Full KS")
    model1 = ot.KernelSmoothing(ot.Epanechnikov(), False, 0, False).build(data)
    pairs(model1.getSample(size_draw), "_model1_" + str(N).zfill(7))
    print("Marginal KS")
    marginals = [ot.KernelSmoothing(ot.Epanechnikov(), False, 0, False).build(data.getMarginal(i)) for i in range(dimension)]
    # Second model: marginal KS + independent copula
    print("Indep. copula")
    model2 = ot.ComposedDistribution(marginals, ot.IndependentCopula(dimension))
    pairs(model2.getSample(size_draw), "_model2_" + str(N).zfill(7))
    # Third model: marginal KS + normal copula
    print("Normal copula")
    model3 = ot.ComposedDistribution(marginals, ot.NormalCopulaFactory().build(data))
    pairs(model3.getSample(size_draw), "_model3_" + str(N).zfill(7))
    # Fourth model: marginal KS + Bernstein copula
    print("Bernstein copula")
    ot.Log.Show(ot.Log.INFO)
    model4 = ot.ComposedDistribution(marginals, ot.BernsteinCopulaFactory().build(data))
    pairs(model4.getSample(size_draw), "_model4_" + str(N).zfill(7))
    print("Model 5")
    alpha = 0.01
    conditioningSet = 4
    learner = otagr.ContinuousPC(data, conditioningSet, alpha)
    learner.setVerbosity(True)
    dag = learner.learnDAG()
    f = open("CBN_" + str(size).zfill(7) + ".dot", "w")
    f.write(dag.toDot());
    f.close()
    print("DAG=", dag)
    factories = [ot.KernelSmoothing(ot.Epanechnikov()), ot.NormalFactory()]
    model5 = otagr.ContinuousBayesianNetworkFactory(factories, dag, alpha, conditioningSet, False).build(data)
    print("sample")
    sample_draw = model5.getSample(size_draw)
    print(sample_draw)
    pairs(sample_draw, "_model5_" + str(size).zfill(7))
    ## print("Model 6")
    ## factories = [ot.KernelSmoothing(ot.Epanechnikov()), ot.NormalCopulaFactory(), ot.BernsteinCopulaFactory()]
    ## copula = otagr.ContinuousBayesianNetworkFactory(factories, dag, alpha, conditioningSet, True).build((data.rank()+1.0)/data.getSize())
    ## marginals = [ot.KernelSmoothing(ot.Epanechnikov(), False, 0, False).build(data.getMarginal(i)) for i in range(data.getDimension())]
    ## model6 = ot.ComposedDistribution(marginals, copula.getCopula())
    ## print("sample")
    ## sample_draw = model6.getSample(size_draw)
    ## print(sample_draw)
    ## pairs(sample_draw, "_model6_" + str(size).zfill(7))

for size in [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    data_ref = ot.Sample.ImportFromTextFile("Standard_coefficients_" + str(10000).zfill(7) + ".csv", ";").getMarginal([i for i in range(12)])
    print("#"*50)
    print("size=", size)
    visu(size, data_ref)
