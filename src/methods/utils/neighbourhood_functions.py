import networkx as nx
import numpy as np
import pandas as pd

def summaries_neighbourhoors_node(node, G_copy, measures):
    G_ego = nx.ego_graph(G_copy, node)
    G_ego.remove_node(node)
    ego_list = list(G_ego.nodes)
    
    list_ego_measures = []
    for n in ego_list:
        list_ego_measures.append(measures[n])

    try:
        min_measure = np.min(list_ego_measures) 
        average_measure = np.mean(list_ego_measures)
        max_measure = np.max(list_ego_measures)
        std_measure = np.std(list_ego_measures)

    except: 
        min_measure = 0
        average_measure = 0
        max_measure = 0
        std_measure = 0
        
    return([min_measure, average_measure, max_measure, std_measure])

def degree_neighbours_node(node, G_copy, G_degree_dict):
    G_ego = nx.ego_graph(G_copy, node)
    G_ego.remove_node(node)
    ego_list = list(G_ego.nodes)
    
    list_ego_degree = []
    for n in ego_list:
        list_ego_degree.append(G_degree_dict[n])

    try:
        min_degree = np.min(list_ego_degree)
        average_degree = np.mean(list_ego_degree)
        max_degree = np.max(list_ego_degree)
        std_degree = np.std(list_ego_degree)

    except: 
        min_degree = 0
        average_degree = 0
        max_degree = 0
        std_degree = 0
        
    return([min_degree, average_degree, max_degree, std_degree])

def GARG_AML_nodeselection_undirected(G_ego_second, node):
    nodes_1 = list(
      nx.ego_graph(G_ego_second, node).nodes)
    nodes_1.remove(node)
    nodes_2 = list(G_ego_second.nodes)
    nodes_2.remove(node)
    for n in nodes_1:
        nodes_2.remove(n)

    # Block order for the undirected score: node + 2nd-order, then 1st-order neighbours.
    nodes_ordered = [node] + nodes_2 + nodes_1
        
    return nodes_1, nodes_2, nodes_ordered

def GARG_AML_nodeselection_directed(G_ego_second, G_ego_second_und, G_ego_second_rev, node):
    nodes_1 = list(
      nx.ego_graph(G_ego_second_und, node).nodes
      )
    nodes_1.remove(node)
    nodes_2 = list(G_ego_second.nodes)

    nodes_2_s = list(
        nx.ego_graph(G_ego_second, node, radius=2).nodes
        )
    
    nodes_2_rs = list(
        nx.ego_graph(G_ego_second_rev, node, radius=2).nodes
    )
    
    nodes_0 = list(
        set(nodes_2).difference(set(nodes_2_s)).difference(set(nodes_2_rs)).difference(set(nodes_1))
    )

    nodes_0 = [node] + nodes_0

    for n in nodes_0:
        nodes_2.remove(n)
    for n in nodes_1:
        nodes_2.remove(n)

    # Block order for the directed score: level 0, then level 1, then level 2.
    nodes_ordered = nodes_0 + nodes_1 + nodes_2
        
    return nodes_0, nodes_1, nodes_2, nodes_ordered

def GARG_AML_nodeselection(G_ego_second, node, directed, G_ego_second_und=None, G_ego_second_rev=None):
    if directed:
        return GARG_AML_nodeselection_directed(G_ego_second, G_ego_second_und, G_ego_second_rev, node)
    else:
        return GARG_AML_nodeselection_undirected(G_ego_second, node)
    
def combine_GARG_AML(G_selection, measures_dict, summary_dict, neigh_degree_dict):
    degree_df = pd.DataFrame(
        dict(
            G_selection.degree()
        ), 
        index = ["degree"]
    ).transpose()
    
    measures_df = pd.DataFrame(
        measures_dict, 
        index = ["GARGAML"]
    ).transpose()

    summary_df = pd.DataFrame(
        summary_dict, 
        index = ["GARGAML_min", "GARGAML_mean", "GARGAML_max", "GARGAML_std"]
    ).transpose()

    neigh_degree_df = pd.DataFrame(
        neigh_degree_dict,
        index = ["degree_min", "degree_mean", "degree_max", "degree_std"]
    ).transpose()
    
    GARG_AML_df = measures_df.merge(
        summary_df,
        left_index = True, 
        right_index = True
    ).merge(
        degree_df, 
        left_index = True, 
        right_index = True
    ).merge(
        neigh_degree_df,
        left_index = True, 
        right_index = True
    )
    
    return(GARG_AML_df)