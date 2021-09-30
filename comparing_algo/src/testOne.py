# -*- coding: utf-8 -*-

import openturns as ot
import otagrum
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

import numpy as np

from pipeline import Pipeline
import graph_utils as gu
# import elidan.hill_climbing as hc

data = ot.Sample.ImportFromTextFile("../data/samples/gaussian/chain_vs/r08/sample01.csv", ',')

# learner = cmiic.ContinuousMIIC(data,
                               # cmode=cmiic.CModeType.Bernstein,
                               # kmode=cmiic.KModeType.Naive)
# learner = otagrum.TabuList(data, 4, 5, 5)
learner = otagrum.ContinuousPC(data, 5, 0.05)
learner.setVerbosity(True)
dag = learner.learnDAG()
gu.write_graph(dag, "output_cpc_gaussian.dot")

# learner = otagrum.TabuList(data, 4, 5, 5)
# learner.setCMode(otagrum.CorrectedMutualInformation.CModeTypes_Bernstein)
# learner.setVerbosity(True)
# dag = learner.learnDAG()
# gu.write_graph(dag, "output_tabulist_bernstein.dot")

# t = hc.hill_climbing(data)
# names = list(data.getDescription())
# gu.write_graph(otagrum.NamedDAG(t[1], names), "output_hc.dot")
# print("Final score hc: ", t[2])
# learner.use3off2()
# pdag = learner.learnMixedStructure()
# dag = learner.learnStructure()

# gnb.showDot(learner._ContinuousMIIC__skeleton.toDot())
# gnb.showDot(pdag.toDot())
# gnb.showDot(dag.toDot())
