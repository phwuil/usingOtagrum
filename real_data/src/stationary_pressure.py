import openturns as ot

class StationaryPressure(ot.OpenTURNSPythonFunction):
    '''
       Allows to compute the stationary pressure of a booster
       given a KL decomposition and a vector of coefficients
    '''
    def __init__(self, studyName = "resultKL.xml"):
        self.resultKL = ot.KarhunenLoeveResult()
        study = ot.Study(studyName)
        study.load()
        study.fillObject("resultKL", self.resultKL)
        mean = ot.Point()
        study.fillObject("mean", mean)
        self.meanMax = mean[-1]
        super(StationaryPressure, self).__init__(self.resultKL.getEigenValues().getSize(), 1)

    def _exec(self, X):
        X = ot.Point(X)
        res = self.resultKL.liftAsSample(X)
        value = res[-1, 0]
        return [self.meanMax + value]

print("Create stationary_pressure")
t0 = time()
stationary_pressure = ot.Function(StationaryPressure())
x = ot.Normal(stationary_pressure.getInputDimension()).getRealization()
print("pressure_max(x)=", stationary_pressure(x))
print("t=", time() - t0, "s")
