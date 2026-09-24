# Stage 1 on the synthetic grid: per-node undirected block measures, written to
# results/<dataset>_GARGAML_undirected_parallel.csv and read back by the tree
# scripts. Run from the repository root; all paths below are root-relative.
import os
import sys
import time
from functools import partial
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import timeit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data.graph_construction import construct_synthetic_graph
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_undirected_measures
from src.utils.runtime import (env_override, select_datasets, echo_config,
                              should_skip, write_csv, log_timing, resolve_results_dir)

# Global variable for worker processes
graph_for_worker = None

def init_worker(graph):
    """
    Initializer for worker processes to set the graph in each subprocess.
    """
    global graph_for_worker
    graph_for_worker = graph

def process_node(node):
    """
    Worker function: computes GARG-AML measures for a single node using the global graph.
    """
    m1, m2, m3, s1, s2, s3 = GARG_AML_node_undirected_measures(
        node, graph_for_worker, include_size=True
    )
    return (node, m1, m2, m3, s1, s2, s3)

def construct_datasets():
    datasets_list = []
    n_nodes_list = [
        100, 
        10000, 
        100000
        ] # Number of nodes in the graph
    m_edges_list = [
        1, 
        2, 
        5
        ] # Number of edges to attach from a new node to existing nodes
    p_edges_list = [
        0.001, 
        0.01
        ] # Probability of adding an edge between two nodes
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
                        p_edges = 0
                        for m_edges in m_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            datasets_list.append(string_name)
                    if generation_method == 'Erdos-Renyi':
                        m_edges = 0
                        for p_edges in p_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            datasets_list.append(string_name)
                    if generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                                datasets_list.append(string_name)
    return datasets_list

datasets = construct_datasets()
directed = False
# Parallelism: use up to 4 or half of CPUs
n_cpu = min(4, cpu_count() // 2)

# Environment overrides, for array jobs: the constants above are the defaults,
# and a task selects its dataset and worker count through the environment
# instead of editing this file. The worker count stays an explicit cap rather
# than an auto-detect, since it determines the runtimes reported below.
datasets = select_datasets(datasets)
n_cpu = env_override("n_cpu", n_cpu, int)
RESULTS_DIR = resolve_results_dir()

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    echo_config(__file__, datasets=datasets, directed=directed, n_cpu=n_cpu, results_dir=RESULTS_DIR)

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        out_path = f"{RESULTS_DIR}/{dataset}_GARGAML_undirected_parallel.csv"
        if should_skip(out_path, dataset):
            continue
        start_time = timeit.default_timer()

        # Load or construct graph
        path = 'data/edge_data_'+dataset+'.csv'
        G = construct_synthetic_graph(path=path, directed = directed)

        G_reduced = graph_community(G)

        nodes = list(G_reduced.nodes)
        print(f"Number of nodes: {len(nodes)} | Using {n_cpu} processes")

        # Initialize pool with graph in each worker
        with Pool(processes=n_cpu, initializer=init_worker, initargs=(G_reduced,)) as pool:
            results = list(tqdm(pool.imap(process_node, nodes), total=len(nodes)))
        # Unpack results
        (
            nodes_out,
            measure_1_list, measure_2_list, measure_3_list,
            size_1_list, size_2_list, size_3_list
        ) = zip(*results)

        elapsed = timeit.default_timer() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")
        # Log timing
        with open(f'{RESULTS_DIR}/time_results_undir.txt', 'a') as f:
            f.write(f"{dataset}: {elapsed:.2f}\n")
        # Per-task timing file beside the shared append, which is a race under
        # an array job. slurm/collect.slurm concatenates these.
        log_timing(dataset, "undirected", elapsed, n_cpu, results_dir=RESULTS_DIR)

        # Save DataFrame
        df = pd.DataFrame({
            "node": nodes_out,
            "measure_1": measure_1_list,
            "measure_2": measure_2_list,
            "measure_3": measure_3_list,
            "size_1": size_1_list,
            "size_2": size_2_list,
            "size_3": size_3_list,
        })
        out_path = f"{RESULTS_DIR}/{dataset}_GARGAML_undirected_parallel.csv"
        write_csv(df, out_path)
        print(f"Results saved to {out_path}")
    print("\nAll datasets processed successfully.")