import os
import sys
import timeit
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

# project paths
DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from src.data.graph_construction import construct_IBM_graph, construct_synthetic_graph
from src.data.bank_views import parse_view, trans_path
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_undirected_measures

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

# List of datasets to process sequentially
# Datasets to process. An entry may be a plain dataset name (the full graph,
# as published) or a task-5 single-bank view "<dataset>_bank<b>", which keeps
# only the transactions booked at bank b. The name is used verbatim for the
# output file, so a view writes results/<dataset>_bank<b>_GARGAML_*.csv beside
# the full-data one. Pick the banks with notebooks/BankObservability.ipynb.
# Task 5 partial-observability views, selected by notebooks/BankObservability.ipynb:
#   012      the largest bank by clients -- 0.512% of all accounts, 1.958% of
#            all transactions
#   top50    the 50 largest banks pooled into one institution -- 10.8% of
#            accounts, 31.4% of transactions, which is the realistic size of a
#            large bank and the only setting with enough positives to evaluate
# The appendix comparison itself is scripts/partial_observability.py; these
# entries exist for running the ordinary tree/boosting pipeline on a view.
# Ordered smallest first, and deliberately not in the order they are
# reported: every entry OVERWRITES its results/ file in place and the range
# is four orders of magnitude -- 12,180 nodes for bank 012, 164,822 for
# top50, 515,080 for HI-Small, 2,054,390 for LI-Large. Stage 2
# (gargaml_tree.py) skips a dataset whose measures are missing rather than
# failing, so an interrupted or trimmed run still leaves a usable results/.
# Comment out what you do not need; LI-Large is the multi-hour job.
datasets = ["HI-Small_bank012", "HI-Small_banktop50",
            "HI-Small", "LI-Large"]
directed = False
# Parallelism: use up to 4 or half of CPUs
n_cpu = min(4, cpu_count() // 2)

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        start_time = timeit.default_timer()

        # Load or construct graph
        base, banks = parse_view(dataset)  # task 5: "HI-Small_bank012" -> ("HI-Small", ["012"])
        if base in ["HI-Small", "LI-Large"]:
            G = construct_IBM_graph(path=trans_path(dataset), directed=directed, banks=banks)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        if banks is not None:
            print(f"Single-bank view of {base}: banks {banks}")

        # Optionally apply community reduction
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
        print(f"Dataset {dataset} completed in {elapsed:.2f} seconds")

        # Log timing
        with open('results/time_results_undir.txt', 'a') as f:
            f.write(f"{dataset}: {elapsed:.2f}\n")

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
        out_path = f"results/{dataset}_GARGAML_undirected.csv"
        df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

    print("\nAll datasets processed successfully.")
