import openturns as ot
import openturns.viewer as otv
from time import time
from stationary_pressure import StationaryPressure

stationary_pressure = ot.Function(StationaryPressure())
ot.ResourceMap.SetAsUnsignedInteger("KernelSmoothing-BinNumber", 1000000000)
# Load the KL coefficients database
print("Load ref. coefficients")
t0 = time()
KL_coefficients_ref = ot.Sample.ImportFromTextFile("Standard_coefficients_1000000.csv", ";")
print("t=", time() - t0, "s")
size = KL_coefficients_ref.getSize()
# Get the corresponding output
pressure_ref = ot.Sample.ImportFromTextFile("pressure_ref.csv", ";")
print("Build and draw ref. pressure distribution")
t0 = time()
dist_ref = ot.KernelSmoothing().build(pressure_ref)
graph = dist_ref.drawPDF()
print("t=", time() - t0, "s")
# Try a naive estimation of the coefficients distribution using
# a multivariate kernel smoothing
print("Build KS coefficients distribution")
t0 = time()
dist_KS = ot.KernelSmoothing().build(KL_coefficients_ref)
print("t=", time() - t0, "s")
# get the corresponding output distribution
small = 200
print("Compute KS pressure")
t0 = time()
coefficients_KS = dist_KS.getSample(small)
coefficients_KS.exportToCSVFile("coefficients_KS.csv")
pressure_KS = stationary_pressure(coefficients_KS)
pressure_KS.exportToCSVFile("pressure_KS.csv")
print("t=", time() - t0, "s")
print("Build and draw KS pressure distribution")
t0 = time()
dist_stationary_pressure_KS = ot.KernelSmoothing().build(pressure_KS)
graph.add(dist_stationary_pressure_KS.drawPDF())
# Try a less naive estimation of the coefficients distribution using
# univariate kernel smoothing for the marginals and a Bernstein copula
print("Build Bernstein coefficients distribution")
t0 = time()
marginals = [ot.HistogramFactory().build(KL_coefficients_ref[0:small].getMarginal(i)) for i in range(KL_coefficients_ref.getDimension())]
copula = ot.EmpiricalBernsteinCopula(KL_coefficients_ref[0:small], KL_coefficients_ref[0:small].getSize())
#copula = ot.BernsteinCopulaFactory().build(KL_coefficients_ref[0:small])
dist_Bernstein = ot.ComposedDistribution(marginals, copula)
print("t=", time() - t0, "s")
# get the corresponding output distribution
print("Compute Bernstein pressure")
t0 = time()
coefficients_Bernstein = dist_Bernstein.getSample(small)
coefficients_Bernstein.exportToCSVFile("coefficients_Bernstein.csv")
pressure_Bernstein = stationary_pressure(coefficients_Bernstein)
pressure_Bernstein.exportToCSVFile("pressure_Bernstein.csv")
print("t=", time() - t0, "s")
print("Build and draw KS pressure distribution")
t0 = time()
dist_stationary_pressure_Bernstein = ot.KernelSmoothing().build(pressure_Bernstein)
graph.add(dist_stationary_pressure_Bernstein.drawPDF())
graph.setColors(["red", "blue", "green"])
graph.setLegends(["Ref", "KS", "Bern"])
print("t=", time() - t0, "s")
print("Q1% ref =", pressure_ref.computeQuantile(0.01))
print("Q1% KS  =", pressure_KS.computeQuantile(0.01))
print("Q1% Bern=", pressure_Bernstein.computeQuantile(0.01))
view = otv.View(graph)
view.save("stationary_pressure_pdf.png")
view.close()
ot.Show(graph)
