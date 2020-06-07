#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import otagrum as otagr

import os
import os.path as path
   
import argparse

import sys
sys.path.append('..')
import loading as load
import data_generation as dg


CLI = argparse.ArgumentParser()
CLI.add_argument("--distribution")
CLI.add_argument("--structure")
CLI.add_argument("--n_sample")
CLI.add_argument("--sample_size")
CLI.add_argument("--correlation")

args = CLI.parse_args()

n_sample = int(args.n_sample)
sample_size = int(args.sample_size)
correlation = args.correlation
distribution = args.distribution
structure = args.structure

if correlation:
    correlation = float(correlation)

if not path.isdir("../../data/samples"):
    os.mkdir("../../data/samples")
    
data_directory = path.join("../../data/samples/", distribution)
data_file_name = structure + "_" + distribution + "_sample"

if not path.isdir(data_directory):
    os.mkdir(data_directory)

Tstruct_file = structure + ".txt"
struct_directory = "../../data/structures/"

data_directory = path.join(data_directory, structure)
if not path.isdir(data_directory):
    os.mkdir(data_directory)

if args.distribution == "gaussian" or args.distribution == "student":
    r_subdir = 'r' + str(args.correlation).replace('.', '')
    data_directory = path.join(data_directory, r_subdir)
    if not path.isdir(data_directory):
        os.mkdir(data_directory)

# If not the good length remove all
ldir = os.listdir(data_directory)
if ldir:
    with open(path.join(data_directory, ldir[0]), 'r') as f:
        if len(f.read().split('\n')) != (sample_size + 2):
            for l in ldir:
                os.remove(path.join(data_directory, l))
            
n_existing_sample = len(os.listdir(data_directory))

Tstruct = load.load_struct(path.join(struct_directory, Tstruct_file))
ndag=otagr.NamedDAG(Tstruct)

for i in range(n_existing_sample, n_sample):
    sample = dg.generate_data(ndag, sample_size, args.distribution, correlation)
    sample.exportToCSVFile(path.join(data_directory, data_file_name) + \
                           '_' + str(i+1).zfill(2) + ".csv", ',')