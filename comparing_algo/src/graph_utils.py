# -*- coding: utf-8 -*-

import otagrum as otagr
import pyAgrum as gum
import numpy as np

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

def create_complete_undigraph(size):
    graph = gum.UndiGraph()
    for i in range(size):
        graph.addNodeWithId(i)
        for j in range(i) :
            graph.addEdge(i, j)
    return graph

def undigraph_to_mixedgraph(undiGraph):
    mixedGraph = gum.MixedGraph()
    for nodeId in undiGraph.nodes():
        mixedGraph.addNodeWithId(nodeId)
    for x,y in undiGraph.edges():
        mixedGraph.addEdge(x,y)
    return mixedGraph

def mixedgraph_to_DAG(mixedGraph):
    dag = gum.DAG()
    for nodeId in mixedGraph.nodes():
        dag.addNodeWithId(nodeId)
    for x,y in mixedGraph.arcs():
        dag.addArc(x,y)
    return dag

def mixedgraph_deepcopy(mixedGraph):
    copy = gum.MixedGraph()
    for nodeId in mixedGraph.nodes():
        copy.addNodeWithId(nodeId)
    for x,y in mixedGraph.arcs():
        copy.addArc(x,y)
    for x,y in mixedGraph.edges():
        copy.addEdge(x,y)
    return copy

def named_dag_to_bn(ndag):
    # DAG to BN
    bn = gum.BayesNet()
    names = ndag.getDescription()
    for name in names:
        bn.add(gum.LabelizedVariable(name))
    for node in range(ndag.getSize()):
        for child in ndag.getChildren(node):
            bn.addArc(names[node], names[child])
    return bn

def dag_to_bn(dag, names):
    # DAG to BN
    bn = gum.BayesNet()
    for name in names:
        bn.add(gum.LabelizedVariable(name))
    for arc in dag.arcs():
        bn.addArc(arc[0], arc[1])
    
    return bn

def fastNamedDAG(dotlike):
    dag = gum.DAG()
    names = []
    for string in dotlike.split(';'):
        if not string:
            continue
        lastId = 0
        notfirst = False
        for substring in string.split('->'):
            forward = True
            for name in substring.split('<-'):
                if name not in names:
                    idVar = dag.addNode()
                    names.append(name)
                else:
                    idVar = names.index(name)
                if notfirst:
                    if forward:
                        dag.addArc(lastId, idVar)
                        forward = False
                    else:
                        dag.addArc(idVar, lastId)
                else:
                    notfirst = True
                    forward = False
                lastId = idVar
    return otagr.NamedDAG(dag, names)
