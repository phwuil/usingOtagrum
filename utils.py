import time
from contextlib import contextmanager

import openturns as ot
import otagrum as otagr
import pyAgrum.lib.notebook as gnb


class timer():
  def __init__(self,name):
    self._name=name
    self._startTime=None
    self._elapsedTime=None

  def __enter__(self):
    self._startTime = time.time()
    self._elapsedTime=None
    return self

  def __exit__(self,type,value,traceback):
    if traceback:
        print("type: {}".format(type))
        print("value: {}".format(value))
        print("traceback: {}".format(traceback))

    s = time.time()-self._startTime
    self._elapsedTime=s
    s, ms = divmod(s, 1)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    elapsedTime = ""
    if h > 0:
      elapsedTime += "{:0>2}:".format(int(h))
    if h > 0 or m > 0:
      elapsedTime += "{:0>2}:".format(int(m))
    elapsedTime += "{:0>2}.{:0>3}s".format(int(s), int(1000 * ms))
    print("+ {} done in {}".format(self._name, elapsedTime))
    self._startTime=None

  def getDuration(self):
    return self._elapsedTime


def learn(data, alpha=0.1, verbose=False):
  with timer("Initiating"):
    learn = otagr.ContinuousPC(data, alpha=alpha)
    learn.setVerbosity(verbose)
    learn.setOptimalPolicy(True)
  with timer("Learning skeleton"):
    sk = learn.getSkeleton()
  print("Nodes : {} , Edges : {}".format(sk.size(), sk.sizeEdges()))
  if sk.size() < 40:
    gnb.showDot(learn.skeletonToDot(sk), size="20", format="png")
  else:
    print(sk.edges())
  with timer("Learning VStructures"):
    mg = learn.getPDAG(sk)
  if mg.size() < 40:
    gnb.showDot(mg.toDot(), size="20", format="png")
  else:
    print(sk.edges())


def generateDataFromCSV(filename, size=-1, dim=-1):  # ,separator=';'):
  with timer("Reading " + filename):
    data = ot.Sample.ImportFromCSVFile(filename)  # ,separator=separator)
    if size == -1:
      size = data.getSize()
    if dim == -1:
      dim = data.getDimension()
    #sample = (data[0:size, 0:dim]).toEmpiricalCopula()
    sample = data[0:size, 0:dim]
    sample = (sample.rank()+1.0)/sample.getSize()
  print("sample : {}x{}".format(sample.getSize(), sample.getDimension()))
  return sample


def generateSampleWithConditionalIndependance1(size=1000):
  dim = 3
  # 0T2 | 1 <=> r12 = r02/r01
  R = ot.CorrelationMatrix(dim)
  R[0, 1] = 0.95
  R[0, 2] = 0.9
  R[1, 2] = R[0, 2] / R[0, 1]
  copula = ot.NormalCopula(R)
  copula.setDescription(["X" + str(i) for i in range(dim)])
  return copula.getSample(size)  # ".exportToCSVFile("conditional_independence_02_1.csv")


def generateSampleWithConditionalIndependance2(size=1000):
  dim = 4
  # 0T3 | 1,2 <=> r23 = (r03*(r12^2-1)+r13*(r01-r02*r12)) / (r01*r12-r02)
  R = ot.CorrelationMatrix(dim)
  R[0, 1] = 0.9
  R[0, 2] = 0.9
  R[0, 3] = 0.9
  R[1, 2] = 0.9
  R[1, 3] = 0.95
  R[2, 3] = (R[0, 3] * (R[1, 2] ** 2 - 1.0) + R[1, 3] * (R[0, 1] - R[0, 2] * R[1, 2])) / (R[0, 1] * R[1, 2] - R[0, 2])
  copula = ot.NormalCopula(R)
  copula.setDescription(["X" + str(i) for i in range(dim)])
  return copula.getSample(size)  # .exportToCSVFile("conditional_independence_03_12.csv")


def generateSampleWithConditionalIndependance3(size=1000):
  dim = 5
  # 0T4 | 1,2,3 <=> r34 = (-r01*r13*r23*r24+r01*r14*r23^2+r02*r13^2*r24-r02*r13*r14*r23-r03*r12*r13*r24-r03*r12*r14
  # *r23+2*r04*r12*r13*r23+r01*r12*r24+r02*r12*r14+r03*r13*r14+r03*r23*r24-r04*r12^2-r04*r13^2-r04*r23^2-r01*r14-r02
  # *r24+r04)/(r01*r12*r23+r02*r12*r13-r03*r12^2-r01*r13-r02*r23+r03)
  R = ot.CorrelationMatrix(dim)
  R[0, 1] = 0.9
  R[0, 2] = 0.9
  R[0, 3] = 0.9
  R[0, 4] = 0.9
  R[1, 2] = 0.9
  R[1, 3] = 0.9
  R[1, 4] = 0.9
  R[2, 3] = 0.9
  R[2, 4] = 0.95
  R[3, 4] = (-R[0, 1] * R[1, 3] * R[2, 3] * R[2, 4] + R[0, 1] * R[1, 4] * R[2, 3] ** 2 + R[0, 2] * R[1, 3] ** 2 * R[
    2, 4] - R[0, 2] * R[1, 3] * R[1, 4] * R[2, 3] - R[0, 3] * R[1, 2] * R[1, 3] * R[2, 4] - R[0, 3] * R[1, 2] * R[
               1, 4] * R[2, 3] + 2 * R[0, 4] * R[1, 2] * R[1, 3] * R[2, 3] + R[0, 1] * R[1, 2] * R[2, 4] + R[0, 2] * R[
               1, 2] * R[1, 4] + R[0, 3] * R[1, 3] * R[1, 4] + R[0, 3] * R[2, 3] * R[2, 4] - R[0, 4] * R[1, 2] ** 2 - R[
               0, 4] * R[1, 3] ** 2 - R[0, 4] * R[2, 3] ** 2 - R[0, 1] * R[1, 4] - R[0, 2] * R[2, 4] + R[0, 4]) / (
                R[0, 1] * R[1, 2] * R[2, 3] + R[0, 2] * R[1, 2] * R[1, 3] - R[0, 3] * R[1, 2] ** 2 - R[0, 1] * R[
              1, 3] - R[0, 2] * R[2, 3] + R[0, 3])
  copula = ot.NormalCopula(R)
  copula.setDescription(["X" + str(i) for i in range(dim)])
  return copula.getSample(size)  # .exportToCSVFile("conditional_independence_04_123.csv")
