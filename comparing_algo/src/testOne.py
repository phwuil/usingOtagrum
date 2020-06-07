# -*- coding: utf-8 -*-

import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

import openturns as ot

from pipeline import Pipeline
import cmiic.continuousMIIC as cmiic
import numpy as np

data = ot.Sample.ImportFromTextFile("../data/samples/dirichle/asia/sample01.csv", ',')[:900]

learner = cmiic.ContinuousMIIC(data,
                               cmode=cmiic.CModeType.Bernstein,
                               kmode=cmiic.KModeType.Naive)
# learner.use3off2()
pdag = learner.learnMixedStructure()
dag = learner.learnStructure()
# print(learner.getIcache())
gnb.showDot(learner._ContinuousMIIC__skeleton.toDot())
gnb.showDot(pdag.toDot())
gnb.showDot(dag.toDot())