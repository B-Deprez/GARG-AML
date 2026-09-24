# Stage 1 on the IBM data: per-node undirected block measures, written to
# results/<dataset>_GARGAML_undirected.csv and read back by scripts/gargaml_tree.py.
# Run from the repository root; all paths below are root-relative.
import os
import sys
import timeit
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from src.data.graph_construction import construct_IBM_graph, construct_synthetic_graph
from src.data.bank_views import parse_view, trans_path
from src.utils.graph_processing import parse_resolution, reduce_graph
from src.methods.GARGAML import GARG_AML_node_undirected_measures
from src.utils.runtime import (env_override, select_datasets, echo_config,
                              should_skip, write_csv, log_timing, resolve_results_dir)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

# Datasets to process, ordered smallest first; comment out what is not needed.
# In a name, a plain dataset is the full graph, "_bank<b>" / "_banktop<k>" a
# single-institution view, "_res<r>" a Louvain resolution and "_nolouvain" no
# reduction at all (see src/utils/graph_processing.parse_resolution). The name
# is used verbatim in the output path, so each entry writes its own measures
# and OVERWRITES that file in place. LI-Large and the no-Louvain arms are the
# expensive runs. Stage 2 (gargaml_tree.py) skips a dataset whose measures are
# missing, so a trimmed or interrupted run still leaves a usable results/.
datasets = ["HI-Small_bank012", "HI-Small_banktop50",
            "HI-Small_res1", "HI-Small_res5",
            "HI-Small",                        # default resolution 10
            "HI-Small_res20", "HI-Small_res50",
            "LI-Large",
            "HI-Small_nolouvain", "LI-Large_nolouvain"]
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
        out_path = f"{RESULTS_DIR}/{dataset}_GARGAML_undirected.csv"
        if should_skip(out_path, dataset):
            continue
        start_time = timeit.default_timer()

        # Load or construct graph
        base, banks = parse_view(dataset)  # "HI-Small_bank012" -> ("HI-Small", ["012"])
        if base in ["HI-Small", "LI-Large"]:
            G = construct_IBM_graph(path=trans_path(dataset), directed=directed, banks=banks)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        if banks is not None:
            print(f"Single-bank view of {base}: banks {banks}")

        # The Louvain setting comes from the dataset name. Passing the dataset
        # makes this stage own the edge-severance record.
        G_reduced = reduce_graph(G, parse_resolution(dataset)[1], dataset)

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
        print(f"Dataset {dataset} completed in {elapsed:.2f} seconds")

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
        out_path = f"{RESULTS_DIR}/{dataset}_GARGAML_undirected.csv"
        write_csv(df, out_path)
        print(f"Results saved to {out_path}")

    print("\nAll datasets processed successfully.")
