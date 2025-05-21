import os
import sys
import timeit
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data.graph_construction import construct_IBM_graph, construct_synthetic_graph
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_directed_measures

# Global variable for worker processes
graph_for_worker = None
graph_for_worker_rev = None
graph_for_worker_undirected = None

def init_worker(graph, graph_rev, graph_undirected):
    """
    Initializer for worker processes to set the graph in each subprocess.
    """
    global graph_for_worker
    global graph_for_worker_rev
    global graph_for_worker_undirected
    graph_for_worker = graph
    graph_for_worker_rev = graph_rev
    graph_for_worker_undirected = graph_undirected

def process_node(node):
    (
        measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22, 
        size_00, size_01, size_02, size_10, size_11, size_12, size_20, size_21, size_22 
    ) = GARG_AML_node_directed_measures(node, graph_for_worker, graph_for_worker_undirected, graph_for_worker_rev, include_size=True)
    return (
        node, 
        measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22, 
        size_00, size_01, size_02, size_10, size_11, size_12, size_20, size_21, size_22
    )

datasets = ["HI-Small", "LI-Large"]
directed = True
# Parallelism: use up to 4 or half of CPUs
n_cpu = min(4, cpu_count() // 2)

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        start_time = timeit.default_timer()

        # Load or construct graph
        if dataset in ["HI-Small", "LI-Large"]:
            path = f"data/{dataset}_Trans.csv"
            G = construct_IBM_graph(path=path, directed=directed)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        
        G_reduced = graph_community(G)

        G_reduced_und = G_reduced.to_undirected()
        G_reduced_rev = G_reduced.reverse(copy=True)

        nodes = list(G_reduced.nodes)
        print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")

        with Pool(processes=n_cpu, initializer=init_worker, initargs=(G_reduced,G_reduced_rev,G_reduced_und,)) as pool:
            results = list(tqdm(pool.imap(process_node, nodes), total=len(nodes)))

        (
            nodes, 
            measure_00_list, measure_01_list, measure_02_list, measure_10_list, measure_11_list, measure_12_list, measure_20_list, measure_21_list, measure_22_list, 
            size_00_list, size_01_list, size_02_list, size_10_list, size_11_list, size_12_list, size_20_list, size_21_list, size_22_list
        ) = zip(*results)

        elapsed = timeit.default_timer() - start_time
        print(f"Dataset {dataset} completed in {elapsed:.2f} seconds")

        # Log timing
        with open("results/time_results_dir.txt", "a") as f:
            f.write(f"{dataset}: {elapsed:.2f}\n")

        # Save DataFrame
        df = pd.DataFrame({
            "node": nodes,
            "measure_00": measure_00_list,
            "measure_01": measure_01_list,
            "measure_02": measure_02_list,
            "measure_10": measure_10_list,
            "measure_11": measure_11_list,
            "measure_12": measure_12_list,
            "measure_20": measure_20_list,
            "measure_21": measure_21_list,
            "measure_22": measure_22_list, 
            "size_00": size_00_list,
            "size_01": size_01_list,
            "size_02": size_02_list,
            "size_10": size_10_list,
            "size_11": size_11_list,
            "size_12": size_12_list,
            "size_20": size_20_list,
            "size_21": size_21_list,
            "size_22": size_22_list
        })
        out_path = f"results/{dataset}_GARGAML_directed.csv"
        df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

    print("\nAll datasets processed successfully.")