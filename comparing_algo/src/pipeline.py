# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('text', usetex=True)  
mpl.rc('font', family='serif')

import pandas as pd
import pyAgrum as gum
import openturns as ot
import otagrum as otagr
import numpy as np
import os
from pathlib import Path
import discretize as dsc
import data_generation as dg
import utils as ut
import graph_utils as gu
import plotting
import time


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
        elif self.method == 'cbic' and not len(parameters) == 3:
            raise TypeError("cbic method takes two arguments ({} given)".format(len(parameters)))
        else:
            self.parameters = parameters
            
    def load_struct(self):
        # with open(self.structure_dir + self.data_structure, 'r') as file:
            # arcs = file.read().replace('\n', '')
        # return gu.fastNamedDAG(arcs)
        dag, names = gu.read_graph(self.structure_dir + self.data_structure + '.dot')
        return otagr.NamedDAG(dag, names)
    
    def write_struct(self, file, ndag):
        gu.write_graph(ndag, file+'.dot')
    
    def loadData(self, path):
        data = ot.Sample.ImportFromTextFile(path, ',')
        return (data.rank()+1)/(data.getSize()+2)
    
    def structuralScoreExists(self, score):
        parameters = 'f' + str(self.begin_size) + 't' + str(self.end_size) + \
                     'np' + str(self.n_points) + 'r' + str(self.n_restart)
        result_name = '_'.join([score, self.method])
        
        if self.method == "cpc":
            result_name += '_' + str(self.parameters['binNumber']) + '_' + \
                           str(self.parameters['alpha']).replace('.', '') + '_'
        elif self.method == "cbic":
            result_name += '_' + str(self.parameters['max_parents']) + '_' + \
                           str(self.parameters['hc_restart']) + '_' + \
                           str(self.parameters['cmode']) + '_'
        elif self.method == "cmiic":
            result_name += '_' + str(self.parameters['cmode']) + "_" + \
                           str(self.parameters['kmode']) + '_'
        elif self.method == "elidan":
            result_name += '_' + str(self.parameters['max_parents']) + '_' + \
                           str(self.parameters['hc_restart']) + '_' + \
                           str(self.parameters['cmode']) + '_'
        elif self.method == "dmiic":
            result_name += '_' + str(self.parameters['dis_method']) + '_' + \
                           str(self.parameters['nbins']) + '_' + \
                           str(self.parameters['threshold']) + '_'
        elif self.method == "lgbn":
            result_name += '_' + str(self.parameters['max_parents']) + '_' + \
                           str(self.parameters['hc_restart']) + '_' 
        else:
            print("Wrong entry for method argument")

        result_name = result_name + parameters
        
        return os.path.exists(os.path.join(self.result_dir,
                                           'scores',
                                           result_name + '.csv'))
        
    
    def loadLearnedStructures(self):
        sizes = np.linspace(self.begin_size, self.end_size, self.n_points, dtype=int)
        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        path = os.path.join(self.result_dir, 'structures',
                            self.method, parameters + '_' + self.result_domain_str)
        list_structures = []
        for i in range(self.n_restart):
            list_by_size = []
            for size in sizes:
                name = 'sample' + str(i+1).zfill(2) + '_size' + str(size)
                dag, var_names = gu.read_graph(os.path.join(path, name + '.dot'))
                ndag = otagr.NamedDAG(dag, var_names)
                list_by_size.append(ndag)
                # with open(os.path.join(path, name + '.dot'), 'r') as file:
                    # arcs = file.read().replace('\n', '')
                    # ndag = gu.fastNamedDAG(arcs)
                    # list_by_size.append(ndag)
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
        Path(os.path.join(self.result_dir, 'scores')).mkdir(parents=True, exist_ok=True)
        
        if self.structuralScoreExists(score):
            return
        
        # Loading true structure
        Tstruct = self.load_struct()
        
        # Learning structures on multiple dataset
        if self.structureFilesExist():
            list_structures = self.loadLearnedStructures()
        else:
            list_structures = self.struct_from_multiple_dataset()

            
        # Setting sizes for which scores are computed
        sizes = np.linspace(self.begin_size, self.end_size, self.n_points, dtype=int)
        # Reshaping sizes for concatenation
        sizes = sizes.reshape(self.n_points, 1)
        
        # Computing structural scores
        scores = ut.structural_score(Tstruct, list_structures, score)
        mean = np.mean(scores, axis=1).reshape((len(scores),1))
        std = np.std(scores, axis=1).reshape((len(scores),1))
        results = np.concatenate((sizes, mean, std), axis=1)

        
        # Writing results
        header = "Size, Mean, Std"
        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        suffix = ''.join(['f', str(self.begin_size), 't', str(self.end_size),
                          'np', str(self.n_points), 'r', str(self.n_restart)])
        res_file_name = '_'.join([score, self.method, parameters, suffix])        
        res_file = res_file_name + '.csv'
        
        print("Writing results in ", os.path.join(self.result_dir, "scores", res_file))
        np.savetxt(os.path.join(self.result_dir, "scores", res_file),
                       results, fmt="%f", delimiter=',', header=header)
        
    def learning(self, sample):
        if self.method == "cpc":
            learner = otagr.ContinuousPC(sample,
                                         self.parameters['binNumber'],
                                         self.parameters['alpha'])
        
            start = time.time()
            ndag = learner.learnDAG()
            end = time.time()
        
            # TTest = otagr.ContinuousTTest(sample, self.parameters['alpha'])
            # jointDistributions = []        
            # for i in range(ndag.getSize()):
                # d = 1+ndag.getParents(i).getSize()
                # if d == 1:
                    # bernsteinCopula = ot.Uniform(0.0, 1.0)
                # else:
                    # K = TTest.GetK(len(sample), d)
                    # indices = [int(n) for n in ndag.getParents(i)]
                    # indices = [i] + indices
                    # bernsteinCopula = ot.EmpiricalBernsteinCopula(sample.getMarginal(indices), K, False)
                # jointDistributions.append(bernsteinCopula)
    
        elif self.method == "cbic":
            #print(sample.getDescription())
            max_parents = self.parameters['max_parents']
            n_restart_hc = self.parameters['hc_restart']
            cmode = self.parameters['cmode']
            learner = otagr.TabuList(sample, max_parents, n_restart_hc, 5)
            learner.setCMode(cmode)
            start = time.time()
            ndag = learner.learnDAG()
            end = time.time()
            #bn = dag_to_bn(dag, Tstruct.names())
            
        elif self.method == "cmiic":
            cmode = self.parameters['cmode']
            kmode = self.parameters['kmode']
            learner = otagr.ContinuousMIIC(sample)
            learner.setCMode(cmode)
            learner.setKMode(kmode)
            learner.setAlpha(self.kalpha)
            # learner.setBeta(self.kbeta)
            start = time.time()
            ndag = learner.learnDAG()
            end = time.time()
            # bn = gu.named_dag_to_bn(ndag)

        elif self.method == "dmiic":
            # learner.setBeta(self.kbeta)
            ndag, start, end = dsc.learnDAG(sample)
            # bn = gu.named_dag_to_bn(ndag)

        elif self.method == "lgbn":
            start = time.time()
            end = time.time()
            
        else:
            print("Wrong entry for method argument !")
        
        return ndag, end-start

    def struct_from_multiple_dataset(self):
        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        struct_path = os.path.join(self.result_dir, 'structures', self.method,
                            parameters + '_' + self.result_domain_str)
        time_path = os.path.join(self.result_dir, 'times', self.method,
                            parameters + '_' + self.result_domain_str)
        Path(struct_path).mkdir(parents=True, exist_ok=True)
        Path(time_path).mkdir(parents=True, exist_ok=True)
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
            times = []
            for size in sizes:
                print("    Learning with", size, "data...")
                sample = data[0:size]
                ndag, deltaT = self.learning(sample)
                print("    Time elapsed: ", deltaT)
                self.write_struct(os.path.join(struct_path, f_name + '_size' + str(size)), ndag)
                list_by_size.append(ndag)
                times.append(deltaT)

            self.write_times(os.path.join(time_path, f_name + '_size' + str(size)), times)
            list_structures.append(list_by_size)
    
        # Transposing result matrix
        list_structures = np.reshape(list_structures, (self.n_restart, self.n_points)).transpose()
        
        return list_structures

    def write_times(self, file_name, times):
        sizes = np.linspace(self.begin_size, self.end_size, self.n_points, dtype=int)
        sizes = sizes.reshape(self.n_points, 1)
        times = np.array(times).reshape(self.n_points, 1)
        results = np.concatenate((sizes, times), axis=1)
        
        # Writing results
        header = "Size, Time"
        # parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        # suffix = ''.join(['f', str(self.begin_size), 't', str(self.end_size),
                          # 'np', str(self.n_points), 'r', str(self.n_restart)])
        # res_file_name = '_'.join([self.method, parameters, suffix])        
        # res_file = res_file_name + '.csv'
        
        print("Writing results in ", file_name + '.csv')
        np.savetxt(file_name + '.csv', results, fmt="%f", delimiter=',', header=header, comments='')

    
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
        # ax.errorbar(sizes, mean, std, capsize=2, elinewidth=1.25, **kwargs)
        ax.errorbar(sizes, mean, std, capsize=4, elinewidth=2.5, **kwargs)
        if error == True:
            plotting.plot_error(sizes, mean, std, alpha_t, ax=ax, color=kwargs['color'])
        
        ax.legend()

    def plotTime(self, fig, ax, error=False, **kwargs):
        Path(os.path.join(self.figure_dir, 'times')).mkdir(parents=True, exist_ok=True)
        times_dir = os.path.join(self.result_dir, "times", self.method)

        parameters = '_'.join([str(v).replace('.','') for v in self.parameters.values()])
        suffix = ''.join(['f', str(self.begin_size), 't', str(self.end_size),
                          'np', str(self.n_points), 'r', str(self.n_restart)])
        times_dir = os.path.join(times_dir, '_'.join([parameters, suffix]))
        print(times_dir)
        
        
        result_files = os.listdir(times_dir)
        results = []
        for rf in result_files:
            results.append(pd.read_csv(os.path.join(times_dir, rf), delimiter=',', index_col=0, header=0))
        df = pd.DataFrame()
        for r in results:
            df = pd.concat([df, r], axis=1, sort=False)
        mean = np.array(df.mean(axis=1))
        std = np.array(df.std(axis=1))
        sizes = np.array(df.index)
        
        # sizes = res[0].astype(int)
        # mean, std = res[1], res[2]

        # mean.plot(ax=ax)
        # ax.errorbar(sizes, mean, std, capsize=2, elinewidth=1.25, **kwargs)
        ax.errorbar(sizes, mean, std, capsize=4, elinewidth=2.5, **kwargs)
        
        ax.legend()
