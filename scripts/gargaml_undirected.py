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
from src.utils.graph_processing import parse_resolution, reduce_graph
from src.methods.GARGAML import GARG_AML_node_undirected_measures
from src.utils.runtime import (env_override, select_datasets, echo_config,
                              should_skip, write_csv, log_timing)

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
#
# Task 4 (R2-M3): the Louvain sensitivity sweep. The setting rides in the
# dataset name -- "_res<r>" for a resolution, "_nolouvain" for no reduction at
# all -- so each arm writes its own measures and a bare name keeps the
# published resolution of 10. See src/utils/graph_processing.parse_resolution.
#
# Cost: HI-Small stage 1 is ~5 min per direction at resolution 10, so the four
# extra resolutions add ~20 min. The **no-Louvain arms are a different order of
# magnitude**, not a slower version of the same thing: the reduction is what
# keeps a second-order ego graph small, and without it HI-Small's reach ~14,900
# accounts, each of which GARG_AML_node_*_measures densifies with
# nx.adjacency_matrix(...).toarray() -- roughly 1.8 GB for one node (measured
# while building task 5's appendix). They are listed last and deliberately:
# expect LI-Large_nolouvain to be infeasible rather than slow, and record that
# outcome, because "what the pre-processing buys" is exactly what R2-M3 asks.
datasets = ["HI-Small_bank012", "HI-Small_banktop50",
            "HI-Small_res1", "HI-Small_res5",
            "HI-Small",                        # the published setting, res 10
            "HI-Small_res20", "HI-Small_res50",
            "LI-Large",
            # No-Louvain arms last -- see the note above.
            "HI-Small_nolouvain", "LI-Large_nolouvain"]
directed = False
# Parallelism: use up to 4 or half of CPUs
n_cpu = min(4, cpu_count() // 2)

# Slurm overrides. The constants above stay the documented defaults: a bare
# `python scripts/...` run behaves exactly as it always has, and an array task
# selects its one dataset through the environment instead of editing this file.
# n_cpu is deliberately still capped at 4 -- worker count changes the runtime
# numbers the scalability figure reports, so it is a knob, not an auto-detect.
datasets = select_datasets(datasets)
n_cpu = env_override("n_cpu", n_cpu, int)

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    echo_config(__file__, datasets=datasets, directed=directed, n_cpu=n_cpu)

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        out_path = f"results/{dataset}_GARGAML_undirected.csv"
        if should_skip(out_path, dataset):
            continue
        start_time = timeit.default_timer()

        # Load or construct graph
        base, banks = parse_view(dataset)  # task 5: "HI-Small_bank012" -> ("HI-Small", ["012"])
        if base in ["HI-Small", "LI-Large"]:
            G = construct_IBM_graph(path=trans_path(dataset), directed=directed, banks=banks)
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        if banks is not None:
            print(f"Single-bank view of {base}: banks {banks}")

        # Task 4: the Louvain setting rides in the dataset name
        # ("HI-Small_res20", "HI-Small_nolouvain"); a bare name keeps
        # the published resolution of 10. Stage 1 owns the severance log.
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
        with open('results/time_results_undir.txt', 'a') as f:
            f.write(f"{dataset}: {elapsed:.2f}\n")
        # Per-task timing beside the legacy append: the shared file records only
        # "<dataset>: <seconds>", which under an array job is both a race and
        # unattributable afterwards. slurm/collect.slurm concatenates these.
        log_timing(dataset, "undirected", elapsed, n_cpu)

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
        write_csv(df, out_path)
        print(f"Results saved to {out_path}")

    print("\nAll datasets processed successfully.")
