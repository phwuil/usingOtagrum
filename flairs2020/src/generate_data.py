#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as path
import os
import utils as ut
import otagrum as otagr
        

# Parameters
n_sample = 10
size = 100000
r = 0.10

# Setting directories location and files
distribution = "gaussian"
structure = "vStruct"

data_directory = path.join("../data/samples/", distribution)
data_file_name = structure + "_" + distribution + "_sample"

Tstruct_file = structure + ".txt"
struct_directory = "../data/structures/"

data_directory = path.join(data_directory, structure)
if not path.isdir(data_directory):
    os.mkdir(data_directory)

if distribution == "gaussian" or distribution == "student":
    r_subdir = 'r' + str(r).replace('.', '')
    data_directory = path.join(data_directory, r_subdir)
    if not path.isdir(data_directory):
        os.mkdir(data_directory)

Tstruct = ut.load_struct(path.join(struct_directory, Tstruct_file))
#Tstruct = gum.fastBN()
ndag=otagr.NamedDAG(Tstruct)

for i in range(n_sample):
    if distribution == "gaussian":
        sample = ut.generate_gaussian_data(ndag, size, r)
    elif distribution == "student":
        sample = ut.generate_student_data(ndag, size, r)
    elif distribution == "dirichlet":
        sample = ut.generate_dirichlet_data(ndag, size)
    else:
        print("Wrong entry for the distribution !")
    sample.exportToCSVFile(path.join(data_directory, data_file_name) + \
                           '_' + str(i+1).zfill(2) + ".csv", ',')