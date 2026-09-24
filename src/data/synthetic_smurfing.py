import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import random

import warnings
warnings.filterwarnings('ignore')

def add_smurfing_patterns_separate(graph, n, num_smurfs):
    # Source, mules and target are all new nodes, so the pattern forms a
    # component disconnected from the original graph.
    for i in range(n):
        num_nodes = graph.vcount()

        try:
            num_nodes_mules = num_smurfs[i]
        except:
            num_nodes_mules = random.randint(2, 10) # two to ten money mules

        for j in range(num_nodes_mules+2): # Add two extra nodes for the source and target
            graph.add_vertex(num_nodes+j)
            graph.vs[num_nodes+j]['new_node'] = True
            graph.vs[num_nodes+j]['laundering'] = True
            graph.vs[num_nodes+j]['separate'] = True
        for k in range(num_nodes_mules):
            graph.add_edge(num_nodes, num_nodes+k+1) # Connect the source to the money mules
            graph.es[graph.get_eid(num_nodes, num_nodes+k+1)]['laundering'] = True
            graph.add_edge(num_nodes+k+1, num_nodes+num_nodes_mules+1) # Connect the money mules to the target
            graph.es[graph.get_eid(num_nodes+k+1, num_nodes+num_nodes_mules+1)]['laundering'] = True
    
    return graph

def add_smurfing_patterns_new_mules(graph, n, num_smurfs, list_nodes):
    # Source and target are existing accounts; the mules between them are
    # new nodes that only ever transact as mules.
    for i in range(n):
        num_nodes = graph.vcount()

        if len(num_smurfs) == 0:
            num_nodes_mules = random.randint(2, 10) # two to ten money mules
        else:
            num_nodes_mules = num_smurfs[i]
        
        # Randomly select source and target node from existing nodes
        source_node = list_nodes.pop()
        target_node = list_nodes.pop()
        graph.vs[source_node]['laundering'] = True
        graph.vs[target_node]['laundering'] = True
        graph.vs[source_node]['new_mules'] = True
        graph.vs[target_node]['new_mules'] = True

        for j in range(num_nodes_mules):
            graph.add_vertex(num_nodes+j)
            graph.vs[num_nodes+j]['new_node'] = True
            graph.vs[num_nodes+j]['laundering'] = True
            graph.vs[num_nodes+j]['new_mules'] = True
        for k in range(num_nodes_mules):
            graph.add_edge(source_node, num_nodes+k) # Connect the source to the money mules
            graph.es[graph.get_eid(source_node, num_nodes+k)]['laundering'] = True
            graph.add_edge(num_nodes+k, target_node) # Connect the money mules to the target
            graph.es[graph.get_eid(num_nodes+k, target_node)]['laundering'] = True
    
    return graph, list_nodes


def add_smurfing_patterns_existing_mules(graph, n, num_smurfs, list_nodes):
    # Source, mules and target are all existing accounts, so the pattern is
    # masked by the normal transactions those accounts already have.
    for i in range(n):
        if len(num_smurfs) == 0:
            num_nodes_mules = random.randint(2, 10) # two to ten money mules
        else:
            num_nodes_mules = num_smurfs[i]
    
        # Randomly select source and target node from existing nodes
        source_node = list_nodes.pop()
        target_node = list_nodes.pop()
        graph.vs[source_node]['laundering'] = True
        graph.vs[target_node]['laundering'] = True
        graph.vs[source_node]['existing_mules'] = True
        graph.vs[target_node]['existing_mules'] = True

        for i in range(num_nodes_mules):
            node = list_nodes.pop()
            graph.vs[node]['laundering'] = True
            graph.vs[node]['existing_mules'] = True
            graph.add_edge(source_node, node)
            graph.es[graph.get_eid(source_node, node)]['laundering'] = True
            graph.add_edge(node, target_node)
            graph.es[graph.get_eid(node, target_node)]['laundering'] = True

    return graph, list_nodes

def add_smurfing_patterns(graph, n, list_nodes, num_smurfs=[] ,type_pattern=''):
    """
    Add smurfing patterns to the original graph
    :param graph: igraph.Graph object
    :param n: number of smurfing patterns to add
    :param list_nodes: pool of existing nodes, each usable in one pattern
    :param num_smurfs: number of smurfs per pattern (empty, one value or n values)
    :param type_pattern: type of smurfing pattern (separate, new_mules, existing_mules)
    :return: the graph with the patterns added, and the remaining node pool
    """

    assert type(num_smurfs) == list, 'Number of smurfs should be a list'
    assert len(num_smurfs) in [0, 1, n], 'Number of smurfs should be either empty, one value or n values, with n number of smurfing patterns'

    if len(num_smurfs) == 1:
        num_smurfs = num_smurfs * n

    if type_pattern == 'separate':
        graph_smurfing = add_smurfing_patterns_separate(graph, n, num_smurfs)
    elif type_pattern == 'new_mules':
        graph_smurfing, list_nodes = add_smurfing_patterns_new_mules(graph, n, num_smurfs, list_nodes)
    elif type_pattern == 'existing_mules':
        graph_smurfing, list_nodes = add_smurfing_patterns_existing_mules(graph, n, num_smurfs, list_nodes)
    else:
        raise ValueError('Invalid type of smurfing pattern')
    
    return graph_smurfing, list_nodes

def create_synthetic_data(n_nodes, m_edges, p_edges, generation_method, n_patterns):
    string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)

    print("=====================================")
    print('Creating synthetic data for:', string_name)
    print('Number of nodes:', n_nodes)
    print('Number of edges:', m_edges)
    print('Probability of edges:', p_edges)
    print('Number of smurfing patterns:', n_patterns)
    print("=====================================")

    if generation_method == 'Barabasi-Albert':
        # Create synthetic Barabasi-Albert graph
        graph = ig.Graph()
        rg = graph.Barabasi(n_nodes, m=m_edges)
    elif generation_method == 'Erdos-Renyi':
        # Create synthetic Erdos-Renyi graph
        graph = ig.Graph()
        rg = graph.Erdos_Renyi(n_nodes, p=p_edges/10) # Keep number of edges smaller
    elif generation_method == 'Watts-Strogatz':
        # Create synthetic Watts-Strogatz graph
        graph = ig.Graph()
        rg = graph.Watts_Strogatz(1, n_nodes, m_edges, p_edges)
    else:
        raise ValueError('Invalid generation method')

    list_nodes = rg.vs.indices
    random.shuffle(list_nodes)

    rg.vs['new_node'] = False 
    rg.vs['laundering'] = False

    rg.vs['separate'] = False
    rg.vs['new_mules'] = False
    rg.vs['existing_mules'] = False

    rg.es['laundering'] = False

    ## Add smurfing patterns
    # All three injection types go into every graph, so that detection can be
    # studied under increasing amounts of masking by normal activity. The node
    # list is passed on, since each node is used in at most one pattern.
    graph_smurfing, list_nodes = add_smurfing_patterns(rg, n_patterns, list_nodes, type_pattern='separate')
    graph_smurfing, list_nodes = add_smurfing_patterns(rg, n_patterns, list_nodes, type_pattern='new_mules')
    graph_smurfing, list_nodes = add_smurfing_patterns(rg, n_patterns, list_nodes, type_pattern='existing_mules')
    graph_smurfing.simplify(combine_edges='max')

    # Only small graphs are plotted; a layout of a larger one is unreadable
    # and slow.
    if graph_smurfing.vcount() < 1000:
        graph_smurfing.vs['color'] = ['red' if x['new_node'] else 'blue' for x in graph_smurfing.vs]
        graph_smurfing.vs['size'] = [10 if x['laundering'] else 5 for x in graph_smurfing.vs]
        graph_smurfing.es['color'] = ['black' if x['laundering'] else 'gray' for x in graph_smurfing.es]

        visual_style = {}
        visual_style["vertex_size"] = graph_smurfing.vs['size']
        visual_style["vertex_color"] = graph_smurfing.vs['color']
        visual_style["edge_color"] = graph_smurfing.es['color']
        visual_style["edge_width"] = 1
        ig.plot(graph_smurfing, **visual_style, target='data/visualisation_network_'+string_name+'.pdf')

    nodes_data = {attr: graph_smurfing.vs[attr] for attr in graph_smurfing.vs.attributes()}
    nodes_df = pd.DataFrame(nodes_data)[['separate', 'new_mules', 'existing_mules', 'laundering']]
    nodes_df.fillna(False, inplace=True)

    source_list = []
    target_list = []
    for edge in graph_smurfing.es:
        source = edge.source
        target = edge.target
        source_list.append(source)
        target_list.append(target)
    
    edges_df = pd.DataFrame({'source': source_list, 'target': target_list})

    print('Number of nodes:', graph_smurfing.vcount())
    print('Number of edges:', graph_smurfing.ecount())
    n_labels = nodes_df['laundering'].sum()
    print('Number of labelled nodes:', n_labels)

    # Node labels and edge list are the two inputs the *_synth.py scripts read.
    nodes_df.to_csv('data/label_data_'+string_name+'.csv', index=False)
    edges_df.to_csv('data/edge_data_'+string_name+'.csv', index=False)


if __name__ == '__main__':
    n_nodes_list = [100, 10000, 100000] # Number of nodes in the graph
    m_edges_list = [1, 2, 5] # Number of edges to attach from a new node to existing nodes
    p_edges_list = [0.001, 0.01] # Probability of adding an edge between two nodes
    generation_method_list = [
        'Barabasi-Albert', 
        'Erdos-Renyi', 
        'Watts-Strogatz'
        ] # Generation method for the graph
    n_patterns_list = [3, 5] # Number of smurfing patterns to add

    for n_nodes in n_nodes_list:
        for n_patterns in n_patterns_list:
            if n_patterns <= 0.06*n_nodes:
                for generation_method in generation_method_list:
                    if generation_method == 'Barabasi-Albert':
                        for m_edges in m_edges_list:
                            # No p_edges for Barabasi-Albert
                            create_synthetic_data(n_nodes, m_edges, 0, generation_method, n_patterns)
                    elif generation_method == 'Erdos-Renyi':
                        for p_edges in p_edges_list:
                            # No m_edges for Erdos-Renyi
                            create_synthetic_data(n_nodes, 0, p_edges, generation_method, n_patterns)
                    elif generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                create_synthetic_data(n_nodes, m_edges, p_edges, generation_method, n_patterns)