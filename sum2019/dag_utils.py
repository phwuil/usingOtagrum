#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum

def find_neighbor(G):
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                newdag = gum.DAG(G)
                if G.existsArc(i,j):
                    newdag.eraseArc(i,j)
                    yield newdag
                    try:
                        newdag.addArc(j,i)
                    except gum.InvalidDirectedCycle:
                        continue
                    yield newdag
                else:
                    try:
                        newdag.addArc(i,j)
                    except gum.InvalidDirectedCycle:
                        continue
                    yield newdag


def create_empty_dag(N):
    dag = gum.DAG()
    for i in range(N):
        dag.addNode()
    return dag


def create_random_dag(N, step=50):
    dag = create_empty_dag(N)
    for i in range(step):
        dag = np.random.choice(list(find_neighbor(dag)))
    return dag