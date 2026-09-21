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
from src.data.bank_views import parse_view, trans_path
from src.utils.graph_processing import parse_resolution, reduce_graph
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
directed = True
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
        
        # Task 4: the Louvain setting rides in the dataset name
        # ("HI-Small_res20", "HI-Small_nolouvain"); a bare name keeps
        # the published resolution of 10. Stage 1 owns the severance log.
        G_reduced = reduce_graph(G, parse_resolution(dataset)[1], dataset)

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