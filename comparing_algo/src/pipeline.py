# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import openturns as ot
import otagrum as otagr
import numpy as np
import os
from pathlib import Path
import elidan.hill_climbing as hc
import data_generation as dg
import utils as ut
import graph_utils as gu
import plotting


class DataGenerator:
    def __init__(self, distribution, structure):
        self.__distribution = distribution
        self.__structure = structure
        
    def generate_data(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # If not the good length remove all
        ldir = os.listdir(self.data_dir)
        for l in ldir:
            with open(os.path.join(self.data_dir, l), 'r') as f:
                if len(f.read().split('\n')) < (self.data_size + 2):
                    os.remove(os.path.join(self.data_dir, l))
                    
        n_existing_sample = len(os.listdir(self.data_dir))
        
        Tstruct = self.load_struct()
        ndag = otagr.NamedDAG(Tstruct)
        
        for i in range(n_existing_sample, self.data_number):
            sample = dg.generate_data(ndag,
                                      self.data_size,
                                      self.data_distribution,
                                      **self.data_parameters)
            data_file_name = "sample" + str(i+1).zfill(2)
            sample.exportToCSVFile(os.path.join(self.data_dir, data_file_name) + ".csv", ',')
    
class Pipeline:
    def __init__(self, method, **parameters):
        # Default location
        self.location = '../'
        self.data_dir_prefix = 'data/samples/'
        self.structure_dir_prefix = 'data/structures/'
        self.result_dir_prefix = 'results/'
        self.figure_dir_prefix = 'figures/'
        
        # Method and its parameters
        self.method = method
        self.checkParameters(parameters)
        self.parameters = parameters
        self.kalpha = 0.01
        self.kbeta = 0.01
        
        # Default data settings
        self.data_distribution = 'gaussian'
        self.data_parameters = {'r':0.8}
        self.data_structure = 'alarm'
        self.data_size = 10000
        self.data_number = 5
        
        
        # Full paths initialization
        self.data_dir = ''
        self.structure_dir = ''
        self.result_dir = ''
        self.figure_dir = ''
        self.setDirs()
        
        # Default result settings
        self.begin_size = 1000
        self.end_size = 10000
        self.n_points = 10
        self.n_restart = self.data_number
        self.result_domain_str = self.setResultDomainStr()
        
    def __repr__(self):
        method = "Method : {}".format(self.method)
        parameters = "Parameters : {}, {}".format(self.parameters[0], self.parameters[1])
        
        distribution = "Data distribution : {}".format(self.data_distribution)
        structure = "Data structure : {}".format(self.data_structure)
    
        return '\n'.join([method, parameters, distribution, structure])
    
    # Magouille
    def setKAlpha(self, kalpha):
        self.kalpha = kalpha
    
    # Magouille
    def setKBeta(self, kbeta):
        self.kbeta = kbeta
    
    # File name management
        
    def setLocation(self, path):
        self.location = path
        self.setDirs()

    def setDirs(self):
        self.setDataDir()
        self.setStructureDir()
        self.setResultDir()
        self.setFigureDir()
        
    def setDataDirPrefix(self, path):
        self.data_dir_prefix = path
        self.setDataDir()
        
    def setStructurePrefix(self, path):
        self.structure_dir_prefix = path
        
    def setResultDirPrefix(self, path):
        self.result_dir_prefix = path
        self.setResultDir()
        
    def setFigureDirPrefix(self, path):
        self.figure_dir_prefix = path
        self.setFigureDir()
        
    def setDataDir(self):
        parameters = self.parametersToPath()
        suffix = os.path.join(self.data_distribution, self.data_structure, parameters)
        self.data_dir = os.path.join(self.location, self.data_dir_prefix, suffix)
    
    def setStructureDir(self):
        self.structure_dir = os.path.join(self.location, self.structure_dir_prefix)
    
    def setResultDir(self):
        parameters = self.parametersToPath()
        suffix = os.path.join(self.data_distribution, self.data_structure, parameters)
        self.result_dir = os.path.join(self.location, self.result_dir_prefix, suffix)
        
    def setFigureDir(self):
        parameters = self.parametersToPath()
        suffix = os.path.join(self.data_distribution, self.data_structure, parameters)
        self.figure_dir = os.path.join(self.location, self.figure_dir_prefix, suffix)
        
    def where(self):
        location = "Location : {}".format(self.location)
        data = "Data : {}".format(self.data_dir_prefix)
        structure = "Structure : {}".format(self.structure_dir_prefix)
        results = "Results : {}".format(self.result_dir_prefix)
        figures = "Figures : {}".format(self.figure_dir_prefix)
        return '\n'.join([location, data, structure, results, figures])
    
    def getFigureDir(self):
        return self.figure_dir
    
    def parametersToPath(self):
        path = ''
        for key, value in self.data_parameters.items():
            string = key + str(value).replace('.', '')
            path = os.path.join(path, string)
        return path
    
    def setDataStructure(self, structure):
        self.data_structure = structure
        self.setDirs()
        
    def setResultDomainStr(self):
        self.result_domain_str = ''.join(['f', str(self.begin_size), 't', str(self.end_size),
                                          'np', str(self.n_points), 'r', str(self.n_restart)])
    
    # def getSizeInfos(self):
    #     return self.begin_size, self.end_size, self.n_points
    
    def generate_data(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # If not the good length remove all
        ldir = os.listdir(self.data_dir)
        for l in ldir:
            with open(os.path.join(self.data_dir, l), 'r') as f:
                if len(f.read().split('\n')) < (self.data_size + 2):
                    os.remove(os.path.join(self.data_dir, l))
                    
        n_existing_sample = len(os.listdir(self.data_dir))
        
        Tstruct = self.load_struct()
        ndag = otagr.NamedDAG(Tstruct)
        
        for i in range(n_existing_sample, self.data_number):
            sample = dg.generate_data(ndag,
                                      self.data_size,
                                      self.data_distribution,
                                      **self.data_parameters)
            data_file_name = "sample" + str(i+1).zfill(2)
            sample.exportToCSVFile(os.path.join(self.data_dir, data_file_name) + ".csv", ',')

    def setDataDistribution(self, distribution, **kwargs):
        self.data_distribution = distribution
        self.data_parameters = dict(kwargs)
        self.setDirs()
    
    def setResultDomain(self, begin_size, end_size, n_points, n_restart):
        self.begin_size = begin_size
        self.end_size = end_size
        self.n_points = n_points
        self.n_restart = n_restart
        self.data_number = n_restart
        self.data_size = end_size
        self.setResultDomainStr()
    
    def setDataSize(self, data_size):
        self.data_size = data_size

    def setNRestart(self, n_restart):
        self.n_restart = n_restart
        
    def checkParameters(self, parameters):
        if self.method == 'cmiic' and not len(parameters) == 2:
            raise TypeError("cmiic method takes two arguments ({} given)".format(len(parameters)))
        elif self.method == 'cpc' and not len(parameters) == 2:
            raise TypeError("cpc method takes two arguments ({} given)".format(len(parameters)))
        elif self.method == 'elidan' and not len(parameters) == 2:
            raise TypeError("elidan method takes two arguments ({} given)".format(len(parameters)))
        else:
            self.parameters = parameters
            
    def load_struct(self):
        with open(self.structure_dir + self.data_structure, 'r') as file:
            arcs = file.read().replace('\n', '')
        return gu.fastNamedDAG(arcs)
    
    def write_struct(self, file, ndag):
        struct_str = ''
        names = ndag.getDescription()
        dag = ndag.getDAG()
        
        for node in range(ndag.getSize()):
            struct_str += names[node] + ';'
            
        for (head,tail) in dag.arcs():
            struct_str += names[head] + "->" + names[tail] + ';'
        
        with open(file, 'w') as f:
            print(struct_str, file=f)
    
    def loadData(self, path):
        data = ot.Sample.ImportFromTextFile(path, ',')
        # return data
        return (data.rank()+1)/(data.getSize()+2)
    
    def structuralScoreExists(self, score):
        parameters = 'f' + str(self.begin_size) + 't' + str(self.end_size) + \
                     'np' + str(self.n_points) + 'r' + str(self.n_restart)
        result_name = '_'.join([score, self.method])
        
        if self.method == "cpc":
            result_name += '_' + str(self.parameters['binNumber']) + '_' + \
                           str(self.parameters['alpha']).replace('.', '') + '_'
        elif self.method == "elidan":
            result_name += '_' + str(self.parameters['max_parents']) + '_' + \
                           str(self.parameters['hc_restart']) + '_'
        elif self.method == "cmiic":
            result_name += '_' + str(self.parameters['cmode']) + "_" + \
                           str(self.parameters['kmode']) + '_'
        else:
            print("Wrong entry for method argument")

        result_name = result_name + parameters
        
        return os.path.exists(os.path.join(self.result_dir,
                                           'scores',
                                           result_name + '.csv'))
        
    
    def loadLearnedStructures(self):
        print("Loading learned structures", flush=True)
        sizes = np.linspace(self.begin_size, self.end_size, self.n_points, dtype=int)
        print("Sizes OK", flush=True)
        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        print("Parameters OK", flush=True)
        path = os.path.join(self.result_dir, 'structures',
                            self.method, parameters + '_' + self.result_domain_str)
        print("Path OK", flush=True)
        list_structures = []
        for i in range(self.n_restart):
            list_by_size = []
            for size in sizes:
                name = 'sample' + str(i+1).zfill(2) + '_size' + str(size)
                with open(os.path.join(path, name), 'r') as file:
                    arcs = file.read().replace('\n', '')
                    print("Arcs: {}".format(arcs))
                    print("Constructing BN")
                    ndag = gu.fastNamedDAG(arcs)
                    print("BN done")
                    list_by_size.append(ndag)
            list_structures.append(list_by_size)
            
        return np.reshape(list_structures, (self.n_restart, self.n_points)).transpose()
    
    def structureFilesExist(self):
        expected_number_file = self.n_restart * self.n_points
        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        path = os.path.join(self.result_dir, 'structures',
                            self.method, parameters + '_' + self.result_domain_str)
        if os.path.exists(path):
            list_file = os.listdir(path)
            if len(list_file) == expected_number_file:
                return True
            else:
                return False
        else:
            return False
    
    def computeStructuralScore(self, score):
        print("Starting to compute scores", flush=True)
        Path(os.path.join(self.result_dir, 'scores')).mkdir(parents=True, exist_ok=True)
        
        if self.structuralScoreExists(score):
            return
        
        # Loading true structure
        Tstruct = self.load_struct()
        print("True structure loaded", flush=True)
        
        # Learning structures on multiple dataset
        if self.structureFilesExist():
            print("File exists", flush=True)
            list_structures = self.loadLearnedStructures()
            print("File loaded", flush=True)
        else:
            print("File doesn't exist", flush=True)
            list_structures = self.struct_from_multiple_dataset()

        print("Structure learned", flush=True)
            
        # Setting sizes for which scores are computed
        sizes = np.linspace(self.begin_size, self.end_size, self.n_points, dtype=int)
        # Reshaping sizes for concatenation
        sizes = sizes.reshape(self.n_points, 1)
        
        # Computing structural scores
        print("About to compute structural score", flush=True)
        scores = ut.structural_score(Tstruct, list_structures, score)
        print("Scores have been computed", flush=True)
        mean = np.mean(scores, axis=1).reshape((len(scores),1))
        std = np.std(scores, axis=1).reshape((len(scores),1))
        results = np.concatenate((sizes, mean, std), axis=1)

        print("Results have been computed", flush=True)
        
        # Writing results
        header = "Size, Mean, Std"
        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        suffix = ''.join(['f', str(self.begin_size), 't', str(self.end_size),
                          'np', str(self.n_points), 'r', str(self.n_restart)])
        res_file_name = '_'.join([score, self.method, parameters, suffix])        
        res_file = res_file_name + '.csv'
        
        print("Writing results in ", os.path.join(self.result_dir, res_file))
        np.savetxt(os.path.join(self.result_dir, "scores", res_file),
                       results, fmt="%f", delimiter=',', header=header)
        
    def learning(self, sample):
        if self.method == "cpc":
            learner = otagr.ContinuousPC(sample,
                                         self.parameters['binNumber'],
                                         self.parameters['alpha'])
        
            ndag = learner.learnDAG()
        
            TTest = otagr.ContinuousTTest(sample, self.parameters['alpha'])
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
    
        elif self.method == "elidan":
            #print(sample.getDescription())
            max_parents = self.parameters['max_parents']
            n_restart_hc = self.parameters['hc_restart']
            copula, dag = hc.hill_climbing(sample, max_parents, n_restart_hc)[0:2]
            #bn = dag_to_bn(dag, Tstruct.names())
            ndag = otagr.NamedDAG(dag, sample.getDescription())
            
        elif self.method == "cmiic":
            cmode = self.parameters['cmode']
            kmode = self.parameters['kmode']
            learner = otagr.ContinuousMIIC(sample)
            learner.setCMode(cmode)
            learner.setKMode(kmode)
            learner.setAlpha(self.kalpha)
            # learner.setBeta(self.kbeta)
            ndag = learner.learnDAG()
            # bn = gu.named_dag_to_bn(ndag)
            
        else:
            print("Wrong entry for method argument !")
        
        return ndag

    def struct_from_multiple_dataset(self):
        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        path = os.path.join(self.result_dir, 'structures', self.method,
                            parameters + '_' + self.result_domain_str)
        Path(path).mkdir(parents=True, exist_ok=True)
        # Looking for which size we learn
        sizes = np.linspace(self.begin_size, self.end_size, self.n_points, dtype=int)
    
        # Looking for all the files in the directory
        files_in_directory = os.listdir(self.data_dir)
        files_in_directory.sort()
        files_in_directory = files_in_directory[:self.n_restart]
        
        list_structures = []
        for f in files_in_directory:
            print("Processing file", f)
            f_name = f.split('.')[0]
            # Loading file f
            data = ot.Sample.ImportFromTextFile(os.path.join(self.data_dir, f), ',')
            data = (data.rank()+1)/(data.getSize()+2)
        
            list_by_size = []
            for size in sizes:
                print("    Learning with", size, "data...")
                sample = data[0:size]
                ndag = self.learning(sample)
                self.write_struct(os.path.join(path, f_name + '_size' + str(size)), ndag)
                list_by_size.append(ndag)

            list_structures.append(list_by_size)
    
        # Transposing result matrix
        list_structures = np.reshape(list_structures, (self.n_restart, self.n_points)).transpose()
        
        return list_structures
    
    def plotScore(self, score, fig, ax, error=False, **kwargs):
        Path(os.path.join(self.figure_dir, 'scores')).mkdir(parents=True, exist_ok=True)

        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        suffix = ''.join(['f', str(self.begin_size), 't', str(self.end_size),
                          'np', str(self.n_points), 'r', str(self.n_restart)])
        res_file_name = '_'.join([score, self.method, parameters, suffix])        
        res_file = res_file_name + '.csv'
        
        res = np.loadtxt(os.path.join(self.result_dir, "scores", res_file), delimiter=',')
        res = res.transpose()
        
        sizes = res[0].astype(int)
        mean, std = res[1], res[2]

        alpha_t = 0.4
        ax.errorbar(sizes, mean, std, capsize=5, elinewidth=2, **kwargs)
        if error == True:
            plotting.plot_error(sizes, mean, std, alpha_t, ax=ax, color=kwargs['color'])
        
        ax.legend()
