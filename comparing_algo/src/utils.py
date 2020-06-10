#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import openturns as ot
import otagrum as otagr
import numpy as np
import pyAgrum as gum
import os
import os.path as path
import elidan.hill_climbing as hc
import graph_utils as gu


def structure_prospecting(structures, index):
    for s in structures[index]:
        print(s.dag())

def structural_score(true_structure, list_structures, score):
    # print(type(true_structure),flush=True)
    # print(type(list_structures[0]),flush=True)
    ref_dag = true_structure.getDAG()
    ref_names = [name for name in true_structure.getDescription()]
    result = []
    for l in list_structures:
        list_result = []
        for s in l: 
            #bn = named_dag_to_bn(s, Tstruct.names())
            test_dag = s.getDAG()
            test_names = [name for name in s.getDescription()]
            sc = gum.StructuralComparator()
            sc.compare(ref_dag, ref_names, test_dag, test_names)
            if score == 'skelP':
                scores = sc.precision_skeleton()
            elif score == 'skelR':
                scores = sc.recall_skeleton()
            elif score == 'skelF':
                scores = sc.f_score_skeleton()
            elif score == 'dagP':
                scores = sc.precision()
            elif score == 'dagR':
                scores = sc.recall()
            elif score == 'dagF':
                scores = sc.f_score()
            elif score == 'hamming':
                scores = sc.shd()
            else:
                print("Wrong entry for argument!")
            
            list_result.append(scores)
        
        result.append(list_result)
    return result

def learning(sample, method, parameters):
    if method == "cpc":
        binNumber, alpha = parameters
        learner = otagr.ContinuousPC(sample, binNumber, alpha)
        
        ndag = learner.learnDAG()
        
        TTest = otagr.ContinuousTTest(sample, alpha)
        jointDistributions = []        
        for i in range(ndag.getSize()):
            d = 1+ndag.getParents(i).getSize()
            if d == 1:
                bernsteinCopula = ot.Uniform(0.0, 1.0)
            else:
                K = TTest.GetK(len(sample), d)
                indices = [int(n) for n in ndag.getParents(i)]
                indices = [i] + indices
                bernsteinCopula = ot.EmpiricalBernsteinCopula(sample.getMarginal(indices), K, False)
            jointDistributions.append(bernsteinCopula)
        
        bn = gu.named_dag_to_bn(ndag)
    
    elif method == "elidan":
        #print(sample.getDescription())
        max_parents, n_restart_hc = parameters
        copula, dag = hc.hill_climbing(sample, max_parents, n_restart_hc)[0:2]
        #bn = dag_to_bn(dag, Tstruct.names())
        bn = gu.dag_to_bn(dag, sample.getDescription())
        
    elif method == "cmiic":
        cmode, kmode = parameters
        print(cmode,kmode)
        learner = cmiic.ContinuousMIIC(sample, cmode, kmode)
        learner.learnMixedStructure()
        dag = learner.learnStructure()
        print(dag)
        bn = gu.dag_to_bn(dag, sample.getDescription())
        print(bn)
    else:
        print("Wrong entry for method argument !")
    
    return bn

def struct_from_multiple_dataset(directory, method, parameters,
                                 start=10, end=1e4, num=10, restart=1):
    # Looking for which size we learn
    sizes = np.linspace(start, end, num, dtype=int)
    
    # Looking for all the files in the directory
    files_in_directory = [f for f in os.listdir(directory) \
                          if path.isfile(path.join(directory, f))]
    files_in_directory.sort()
    files_in_directory = files_in_directory[:restart]
    
    list_structures = []
    for f in files_in_directory:
        print("Processing file", f)
        # Loading file f
        data = ot.Sample.ImportFromTextFile(path.join(directory, f), ',')
        
        list_by_size = []
        for size in sizes:
            print("    Learning with", size, "data...")
            sample = data[0:size]
            bn = learning(sample, method, parameters)
            list_by_size.append(bn)

        list_structures.append(list_by_size)

    # Transposing result matrix
    list_structures = np.reshape(list_structures, (len(files_in_directory), num)).transpose()
    return list_structures

