import os
import sys
import time
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_undirected_measures

dataset = "HI-Small"
path = "data/"+dataset+"_Trans.csv"
directed = False

G = construct_IBM_graph(path=path, directed = directed)
G_reduced = graph_community(G)

# Define a function to process each node
def process_node(node):
    measure_1, measure_2, measure_3, size_1, size_2, size_3 = GARG_AML_node_undirected_measures(node, G_reduced, include_size=True)
    return (node, measure_1, measure_2, measure_3, size_1, size_2, size_3)

if __name__ == '__main__':
    # Get the list of nodes
    nodes = list(G_reduced.nodes)

    print("Here we go")
    start_time = time.time()
    # Use multiprocessing to process nodes in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_node, nodes), total=len(nodes)))

    # Unpack the results
    nodes, measure_1_list, measure_2_list, measure_3_list, size_1_list, size_2_list, size_3_list = zip(*results)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total time taken: {elapsed_time:.2f} seconds")

    data_dict = {
        "node": nodes,
        "measure_1": measure_1_list,
        "measure_2": measure_2_list,
        "measure_3": measure_3_list,
        "size_1": size_1_list,
        "size_2": size_2_list,
        "size_3": size_3_list
    }

    df = pd.DataFrame(data_dict)
    df.to_csv("results/"+dataset+"_GARGAML_undirected.csv", index=False)
