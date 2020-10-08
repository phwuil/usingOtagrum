# -*- coding: utf-8 -*-

import otagrum as otagr
import pyAgrum as gum
import numpy as np
import pydotplus as dot

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

def dag_to_cpdag(dag):
    cpdag = gum.MixedGraph()

    for node in dag.nodes():
        cpdag.addNodeWithId(node)

    for arc in dag.arcs():
        cpdag.addArc(arc[0], arc[1])

    v = []
    while True:
        v.clear()
        for x in dag.topologicalOrder():
            for y in cpdag.children(x):
                if not strongly_protected(cpdag, x, y):
                    v.append((x,y))
        for arc in v:
            cpdag.eraseArc(arc[0], arc[1])
            cpdag.addEdge(arc[0], arc[1])
        if len(v) <=0:
            break

    return cpdag

def strongly_protected(mixed_graph, node1, node2):
    if len(mixed_graph.parents(node1) - mixed_graph.parents(node2)) > 0:
        return True

    cs = set()
    for node3 in mixed_graph.parents(node2):
        if node3 == node1:
            continue
        if not mixed_graph.existsEdge(node3, node1):
            return True
        else:
            cs.add(node3)

    ss = cs.copy()
    if len(cs) < 2:
        return False
    else:
        for i in cs:
            ss = ss - mixed_graph.neighbours(i)
            if len(ss) < 2:
                return False
        return True

def write_graph(graph, file_name="output.dot"):
    with open(file_name, 'w') as fo:
        fo.write(graph.toDot())

def read_graph(file_name):
    print("Loading file {}".format(file_name))
    dot_graph = dot.graph_from_dot_file(file_name)
    isUndiGraph = False

    # Cleaning nodes
    for node in dot_graph.get_nodes():
        name = node.get_name()
        if name in ['node', 'edge']:
            if name == 'edge':
                if node.get_attributes() and node.get_attributes()['dir'] == 'none':
                    isUndiGraph = True
            dot_graph.del_node(node)

    # Getting node names
    node_name_map = {}
    for i,node in enumerate(dot_graph.get_nodes()):
        node_name_map[node.get_name()] = i
    nodeId = max(node_name_map.values()) + 1
    for edge in dot_graph.get_edges():
        source = edge.get_source()
        destination = edge.get_destination()
        if source not in node_name_map.keys():
            node_name_map[source] = nodeId
            nodeId += 1
        if destination not in node_name_map.keys():
            node_name_map[destination] = nodeId
            nodeId += 1

    edges = []
    arcs = []
    for edge in dot_graph.get_edges():
        if (isUndiGraph or
                (edge.get_attributes() and edge.get_attributes()['dir'] == 'none')):
            edges.append(gum.Edge(node_name_map[edge.get_source()],
                                  node_name_map[edge.get_destination()]))
        else:
            arcs.append(gum.Arc(node_name_map[edge.get_source()],
                                node_name_map[edge.get_destination()]))

    if not edges: # DAG
        graph = gum.DAG()
        for node_name in node_name_map:
            graph.addNodeWithId(node_name_map[node_name])
        for arc in arcs:
            graph.addArc(arc.tail(), arc.head())
            
    elif not arcs: # UndiGraph
        graph = gum.UndiGraph()
        for node_name in node_name_map:
            graph.addNodeWithId(node_name_map[node_name])
        for edge in edges:
            graph.addEdge(edge.first(), edge.second())

    else: # MixedGraph
        graph = gum.MixedGraph()
        for node_name in node_name_map:
            graph.addNodeWithId(node_name_map[node_name])
        for edge in edges:
            graph.addEdge(edge.first(), edge.second())
        for arc in arcs:
            graph.addArc(arc.tail(), arc.head())

    # Since python3.7, dict are insertion ordered so
    # just returning values should be fine but we never know !
    return graph, list(node_name_map.keys())
