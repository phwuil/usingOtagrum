#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as path
import os
import utils as ut
import otagrum as otagr
        
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--distribution")
CLI.add_argument("--structure")
CLI.add_argument("--n_sample")
CLI.add_argument("--sample_size")
CLI.add_argument("--correlation")

args = CLI.parse_args()

n_sample = int(args.n_sample)
sample_size = int(args.sample_size)
correlation = float(args.correlation)

data_directory = path.join("../data/samples/", args.distribution)
data_file_name = args.structure + "_" + args.distribution + "_sample"

Tstruct_file = args.structure + ".txt"
struct_directory = "../data/structures/"

data_directory = path.join(data_directory, args.structure)
if not path.isdir(data_directory):
    os.mkdir(data_directory)

if args.distribution == "gaussian" or args.distribution == "student":
    r_subdir = 'r' + str(args.correlation).replace('.', '')
    data_directory = path.join(data_directory, r_subdir)
    if not path.isdir(data_directory):
        os.mkdir(data_directory)

n_existing_sample = len(os.listdir(data_directory))

Tstruct = ut.load_struct(path.join(struct_directory, Tstruct_file))
#Tstruct = gum.fastBN()
ndag=otagr.NamedDAG(Tstruct)

for i in range(n_existing_sample, n_sample):
    if args.distribution == "gaussian":
        sample = ut.generate_gaussian_data(ndag, sample_size, correlation)
    elif args.distribution == "student":
        sample = ut.generate_student_data(ndag, sample_size, correlation)
    elif args.distribution == "dirichlet":
        sample = ut.generate_dirichlet_data(ndag, sample_size)
    else:
        print("Wrong entry for the distribution !")
    sample.exportToCSVFile(path.join(data_directory, data_file_name) + \
                           '_' + str(i+1).zfill(2) + ".csv", ',')