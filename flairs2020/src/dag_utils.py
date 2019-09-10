#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyAgrum as gum

def find_neighbor(G, max_parents=4):
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                newdag = gum.DAG(G)
                # If arcs (i,j) exists we delete it or we reverse it
                if G.existsArc(i,j):
                    newdag.eraseArc(i,j)
                    yield newdag
                    if len(G.parents(i)) < max_parents:
                        try:
                            newdag.addArc(j,i)
                        except gum.InvalidDirectedCycle:
                            continue
                        yield newdag
                # Else if it doesn't exist and this node doesn't have
                # more parents thant max_parents, we add it
                elif len(G.parents(j)) < max_parents:
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


def create_random_dag(N, max_parents=4, step=50):
    dag = create_empty_dag(N)
    for i in range(step):
        neighbors = list(find_neighbor(dag, max_parents))
        if neighbors:
            dag = np.random.choice(neighbors)
    return dag

def max_parents(dag):
    max_nodes = []
    max_num_par = 0
    for node in dag.nodes():
        num_par = len(dag.parents(node))
        if max_num_par < num_par:
            max_nodes = []
            max_nodes.append(node)
            max_num_par = num_par
        elif max_num_par == num_par:
            max_nodes.append(node)

    return max_nodes, max_num_par

    