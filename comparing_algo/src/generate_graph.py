#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# It would be nice to put this in the pipeline
import pyAgrum as gum
import otagrum as otagr
import graph_utils as gu

from pathlib import Path

step = 1
start_size = 7
end_size = 7
restart = 1

density = 1.2

# Setting directories location and files
directory = Path("../data/structures/generated/")
directory.mkdir(parents=True, exist_ok=True)

generator = gum.BNGenerator()
gum.initRandom(10)

for r in range(restart):
    for i in range(start_size, end_size+1, step):
        print("Number of node :", i, flush=True)
        file_name = "size_" + str(i).zfill(3) + "_" + str(r+1).zfill(2)

        n_arc = int( density*(i-1) )
        bn = generator.generate(i, n_arc)
        ndag = otagr.NamedDAG(bn.dag(), bn.names())
        gu.write_graph(ndag, directory.joinpath(file_name + '.dot'))
