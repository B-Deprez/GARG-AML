import pandas as pd
import networkx as nx
import numpy as np

from tqdm import tqdm

from .utils.measure_functions_undirected import *
from .utils.measure_functions_directed import *
from .utils.neighbourhood_functions import summaries_neighbourhoors_node, degree_neighbours_node, GARG_AML_nodeselection, combine_GARG_AML

def GARG_AML_node_directed_measures(node, G_copy, G_copy_und, G_copy_rev, include_size = False):
    G_ego_second_und = nx.ego_graph(G_copy_und, node, 2) #Use both incoming and outgoing edges
    G_ego_second = nx.subgraph(G_copy, G_ego_second_und.nodes)
    G_ego_second_rev = nx.ego_graph(G_copy_rev, node, 2) #Look at the reverse graph to get the incoming edges

    nodes_0, nodes_1, nodes_2, nodes_ordered = GARG_AML_nodeselection(G_ego_second, node, directed = True, G_ego_second_und = G_ego_second_und, G_ego_second_rev = G_ego_second_rev)

    adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered).toarray()

    size_0 = len(nodes_0)
    size_1 = len(nodes_1)
    size_2 = len(nodes_2)

    measure_00, size_00 = measure_00_function(adj_full, size_0)
    measure_01, size_01 = measure_01_function(adj_full, size_0, size_1)
    measure_02, size_02 = measure_02_function(adj_full, size_0, size_1, size_2)
    measure_10, size_10 = measure_10_function(adj_full, size_0, size_1)
    measure_11, size_11 = measure_11_function(adj_full, size_0, size_1)
    measure_12, size_12 = measure_12_function(adj_full, size_0, size_1, size_2)
    measure_20, size_20 = measure_20_function(adj_full, size_0, size_1, size_2)
    measure_21, size_21 = measure_21_function(adj_full, size_0, size_1)
    measure_22, size_22 = measure_22_function(adj_full, size_0, size_1, size_2)

    if include_size:
        return(
            measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22, 
            size_00, size_01, size_02, size_10, size_11, size_12, size_20, size_21, size_22
        )
    else:
        return(measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22)

def GARG_AML_node_directed(node, G_copy, G_copy_und, G_copy_rev):
    measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22 = GARG_AML_node_directed_measures(node, G_copy, G_copy_und, G_copy_rev)

    # Based on the given direction
    measure_high = np.mean([measure_01, measure_12])
    measure_low = np.mean([measure_10, measure_21, measure_00, measure_02, measure_11, measure_20, measure_22])
    measure = measure_high - measure_low

    # Based on the transpose
    measure_high_transpose = np.mean([measure_10, measure_21])
    measure_low_transpose = np.mean([measure_01, measure_12, measure_00, measure_20, measure_11, measure_02, measure_22])
    measure_transpose = measure_high_transpose - measure_low_transpose

    measure_final = np.max([measure, measure_transpose])
    
    return(measure_final)

def GARG_AML_node_undirected_measures(node, G_copy, include_size = False):
    G_ego_second = nx.ego_graph(G_copy, node, 2)
    
    # nodes_ordered are the nodes ordered as node, 2nd order and 1st order neighbours
    nodes_1, nodes_2, nodes_ordered = GARG_AML_nodeselection(G_ego_second, node, directed = False)
    
    adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered).toarray()
    
    size_second = len(nodes_2)
    size_first = len(nodes_1)

    piece_1_dim = [size_second + 1, size_second + 1]
    piece_2_dim = [size_first, size_second + 1]
    piece_3_dim = [size_first, size_first]
    
    measure_1, size_1 = measure_1_function(piece_1_dim, adj_full)
    measure_2, size_2 = measure_2_function(piece_1_dim, piece_2_dim, adj_full)
    measure_3, size_3 = measure_3_function(piece_1_dim, piece_2_dim, piece_3_dim, adj_full)

    if include_size:
        return(measure_1, measure_2, measure_3, size_1, size_2, size_3)
    else:
        return(measure_1, measure_2, measure_3)
    
def GARG_AML_node_undirected(node, G_copy):
    measure_1, measure_2, measure_3 = GARG_AML_node_undirected_measures(node, G_copy)
    measure = measure_2 - (measure_1 + measure_3)/2
    return(measure)

def GARG_AML_node(node, G_copy, G_copy_und=None, G_copy_rev=None, directed = False):
    if directed:
        return(GARG_AML_node_directed(node, G_copy, G_copy_und, G_copy_rev))
    else:
        return(GARG_AML_node_undirected(node, G_copy))
    

def GARG_AML(G_reduced): # The method works with a pre-processed graph. G_reduced is the graph with the degree cutoff applied.
    directed = nx.is_directed(G_reduced)
    if directed:
        G_reduced_und = G_reduced.to_undirected()
        G_reduced_rev = G_reduced.reverse(copy=True)
    
    G_degree_dict = dict(G_reduced.degree())
    
    GARG_AML_values = dict()
    
    nodes = list(G_reduced.nodes)
    for node in tqdm(nodes):
        if directed:
            GARG_AML_node_value = GARG_AML_node(node, G_reduced, G_copy_und=G_reduced_und, G_copy_rev=G_reduced_rev, directed=directed)
        else:
            GARG_AML_node_value = GARG_AML_node(node, G_reduced, directed=directed)
        GARG_AML_values[node] = GARG_AML_node_value
        
    summaries_neighbours = dict()
    degree_neighbours = dict()
    
    for node in nodes:
        summaries_neighbours[node] = summaries_neighbourhoors_node(node, G_reduced, GARG_AML_values)
        degree_neighbours[node] = degree_neighbours_node(node, G_reduced, G_degree_dict)
    
    GARG_AML_df = combine_GARG_AML(G_reduced, GARG_AML_values, summaries_neighbours, degree_neighbours)
    
    return(GARG_AML_df)

