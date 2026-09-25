"""
Partial observability: GARG-AML under a single-institution view.

A bank sees only the transactions with one of its own customers on them, so
the second-order neighbourhood GARG-AML is computed from may not be
observable to it. This script measures what that costs, on the pure
GARG-AML score rather than on the tree/boosting extensions: the score needs
no training, so it survives the small positive counts a single institution
has, where the supervised grid does not.

The comparison
--------------
For every client of the institution, the score is computed twice:

=========  ==================================================================
``full``   the raw transaction graph, everything observable
``view``   the raw graph of the visible transactions only, i.e. those
           booked at the institution on either side
=========  ==================================================================

The Louvain reduction is switched off on both sides. With it on, full and
view would be reduced by different partitions, and the degradation could not
be attributed to the missing edges rather than to a changed partition. The
``full`` column here is therefore not the same quantity as the main results
tables; it is a baseline computed for this comparison.

Only the institution's clients are scored, on both sides. That is the
population a bank alerts on, and it keeps the run short: the no-Louvain
second-order neighbourhoods are large.

Why the comparison is paired
----------------------------
A bank sees every transaction of its own clients, so each client's
first-order neighbourhood is identical under both regimes -- degree is
preserved exactly, and only N2 degrades. The same accounts, the same labels
and the same degree subgroups therefore appear on both sides, and the only
thing that varies is what the score could see. ``degree_differs`` in the
output counts the exceptions, which can only be accounts held at two banks.

Ties
----
The score is heavily tied on a single institution's clients, so "the top 50"
is not uniquely defined and a top-K overlap between the two regimes is not
interpretable on its own. Every ranking metric is therefore written with its
``ties@K`` beside it, and the metrics are additionally reported on the
``degree >= MIN_DEGREE`` subgroup, where the block structure is not
degenerate.

Outputs
-------
  * ``results/<view>_partial_observability_accounts.csv`` -- one row per
    (account, direction): both scores, degree, N1/N2 under both regimes and
    the label propensities.
  * ``results/<view>_partial_observability_metrics.csv`` -- tidy metrics,
    one row per (direction, regime, cut-off, target, subgroup, metric, K).
"""

import os
import sys
import timeit
from multiprocessing import Pool, cpu_count

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data.bank_views import (bank_clients, filter_transactions, patterns_path,
                                 resolve_banks, trans_path, view_name)
from src.data.graph_construction import construct_IBM_graph
from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.methods.GARGAML import (GARG_AML_node_directed_measures,
                                 GARG_AML_node_undirected_measures)
from src.methods.gargaml_scores import define_gargaml_scores
from src.utils.evaluation import evaluate_scores, metric_records, metrics_frame
from src.utils.naming import gargaml_key
from src.utils.runtime import (env_override, select_datasets, echo_config, as_list,
                              resolve_results_dir)

DATASET = "HI-Small"

# Each entry is a bank identifier, or a "top<k>" group standing for the k
# banks with the most clients (see src/data/bank_views.py). The largest
# single bank in HI-Small and a pooled group of the fifty largest are both
# reported: no single bank holds enough positive clients on its own.
INSTITUTIONS = ["012", "top50"]

DIRECTIONS = [False, True]
SCORE_TYPE = "weighted_average"

# 0.0 means "involved in at least one laundering transaction". A single
# institution has too few positive clients at the higher cut-offs, so this
# experiment stays on the two most populated ones rather than the full sweep
# in src/utils/evaluation.py.
CUT_OFFS = [0.0, 0.1]
TARGET_COLUMNS = ["Is Laundering", "GATHER-SCATTER", "SCATTER-GATHER"]

# Below this degree the block structure is degenerate: an ego graph with one
# or two neighbours has near-empty blocks, and the score collapses onto a
# handful of values. Metrics are reported on the whole client base and on
# this subgroup. Degree is preserved by the view, so the subgroup is the same
# set under both regimes.
MIN_DEGREE = 3

# Includes sizes below the module default [50, 100, 500, 1000]: a single
# bank's client base is small enough that K = 1000 is a third of it rather
# than an alert queue.
ALERT_SIZES = [10, 25, 50, 100, 500, 1000]

n_cpu = min(4, cpu_count() // 2)

# Slurm overrides. One institution per array task: 'top50' is far more
# expensive than '012'.
DATASET = env_override("dataset", DATASET)
INSTITUTIONS = env_override("institutions", INSTITUTIONS, as_list)
n_cpu = env_override("n_cpu", n_cpu, int)
RESULTS_DIR = resolve_results_dir()

UNDIRECTED_COLUMNS = ["node",
                      "measure_1", "measure_2", "measure_3",
                      "size_1", "size_2", "size_3"]
DIRECTED_COLUMNS = ["node"] + \
    ["measure_"+str(i)+str(j) for i in range(3) for j in range(3)] + \
    ["size_"+str(i)+str(j) for i in range(3) for j in range(3)]

graph_for_worker = None
graph_for_worker_und = None
graph_for_worker_rev = None


def init_worker(graph, graph_und, graph_rev):
    """Sets the graph globals in each worker subprocess."""
    global graph_for_worker
    global graph_for_worker_und
    global graph_for_worker_rev
    graph_for_worker = graph
    graph_for_worker_und = graph_und
    graph_for_worker_rev = graph_rev


def process_node_undirected(node):
    m1, m2, m3, s1, s2, s3 = GARG_AML_node_undirected_measures(
        node, graph_for_worker, include_size=True
    )
    return (node, m1, m2, m3, s1, s2, s3)


def process_node_directed(node):
    measures = GARG_AML_node_directed_measures(
        node, graph_for_worker, graph_for_worker_und, graph_for_worker_rev,
        include_size=True
    )
    return (node,) + tuple(measures)


def measure_clients(G, clients, directed):
    """Block measures for ``clients`` only, on ``G`` exactly as given.

    ``G`` is not Louvain-reduced here -- see the module docstring. Only the
    requested nodes are measured, which is what keeps this affordable on the
    un-severed graph.
    """
    if directed:
        initargs = (G, G.to_undirected(), G.reverse(copy=True))
        worker, columns = process_node_directed, DIRECTED_COLUMNS
    else:
        initargs = (G, None, None)
        worker, columns = process_node_undirected, UNDIRECTED_COLUMNS

    with Pool(processes=n_cpu, initializer=init_worker, initargs=initargs) as pool:
        results = list(tqdm(pool.imap(worker, clients), total=len(clients)))

    return pd.DataFrame(results, columns=columns)


def client_scores(G, clients, directed):
    """The aggregated GARG-AML score per client, indexed by account."""
    measures = measure_clients(G, clients, directed)
    return define_gargaml_scores(measures, directed, score_type=SCORE_TYPE)


def neighbourhood_sizes(G, clients):
    """|N1| and |N2| per client -- the observability diagnostic.

    N1 is the control: it is preserved exactly by a view, because every
    transaction of a client is visible to that client's own bank. N2 is
    what partial observability destroys, and what the block structure is
    actually computed from.
    """
    n1, n2 = [], []
    for node in clients:
        distances = nx.single_source_shortest_path_length(G, node, cutoff=2).values()
        n1.append(sum(1 for d in distances if d == 1))
        n2.append(sum(1 for d in distances if d == 2))
    return pd.Series(n1, index=clients), pd.Series(n2, index=clients)


def evaluate_regimes(accounts, labels, direction, context):
    """Metrics for both regimes, every cut-off/target, both subgroups."""
    subgroups = {
        "all": pd.Series(True, index=accounts.index),
        "degree>=" + str(MIN_DEGREE): accounts["degree"] >= MIN_DEGREE,
    }

    records = []
    for subgroup, keep in subgroups.items():
        for cutoff in CUT_OFFS:
            for target in TARGET_COLUMNS:
                y = (labels.loc[accounts.index, target] > cutoff).astype(int)[keep]

                for regime in ["full", "view"]:
                    score = accounts.loc[keep, "GARGAML_" + regime]
                    cell = dict(context, direction=direction, regime=regime,
                                subgroup=subgroup, cutoff=cutoff, target=target)
                    try: # a cell with no positives has no AUC; reported, not dropped
                        metrics = evaluate_scores(y.values, score.values,
                                                  alert_sizes=ALERT_SIZES)
                        status = "ok"
                    except Exception as exc:
                        print("    skipped "+target+" @"+str(cutoff)+" ("+subgroup+
                              ", "+regime+"): "+repr(exc))
                        metrics, status = {}, "skipped: "+str(exc)

                    records += metric_records(metrics, status=status,
                                              n_test=int(keep.sum()),
                                              n_pos=int(y.sum()), **cell)
    return records


def run_institution(banks_spec):
    """The whole experiment for one institution."""
    path = trans_path(DATASET)
    banks = resolve_banks(banks_spec, path)
    view = view_name(DATASET, banks_spec)

    print("\n=== "+view+": "+str(len(banks))+" bank(s) ===")
    start_time = timeit.default_timer()

    # Coverage: the "x% of clients, y% of transactions" statement.
    transactions = pd.read_csv(path, usecols=["From Bank", "Account",
                                              "To Bank", "Account.1"],
                               dtype={"From Bank": str, "To Bank": str,
                                      "Account": str, "Account.1": str})
    visible = filter_transactions(transactions, banks)
    clients = sorted(bank_clients(visible, banks))
    n_accounts = len(set(transactions["Account"]) | set(transactions["Account.1"]))

    print("  clients      "+str(len(clients))+" ("+
          format(100*len(clients)/n_accounts, ".3f")+"% of all accounts)")
    print("  visible tx   "+str(len(visible))+" ("+
          format(100*len(visible)/len(transactions), ".3f")+"% of all transactions)")

    # Labels from the visible transactions: a client's propensity is the same
    # either way (the bank sees all of its clients' transactions), and this
    # reads ~2% of the rows instead of all of them.
    transactions_extended, pattern_columns = define_ML_labels(
        path_trans=path, path_patterns=patterns_path(DATASET), banks=banks
    )
    labels, _, _ = summarise_ML_labels(transactions_extended, pattern_columns)
    labels = labels.loc[clients]

    frames = []
    for directed in DIRECTIONS:
        str_directed = "directed" if directed else "undirected"
        print("\n  --- "+str_directed+" ---")

        graphs = {
            "full": construct_IBM_graph(path=path, directed=directed),
            "view": construct_IBM_graph(path=path, directed=directed, banks=banks),
        }
        print("  full graph: "+str(graphs["full"].number_of_nodes())+" nodes, "
              +str(graphs["full"].number_of_edges())+" edges | view: "
              +str(graphs["view"].number_of_nodes())+" nodes, "
              +str(graphs["view"].number_of_edges())+" edges")

        accounts = pd.DataFrame(index=pd.Index(clients, name="account"))
        accounts["direction"] = str_directed

        for regime, G in graphs.items():
            print("  scoring "+str(len(clients))+" clients on the "+regime+" graph")
            scores = client_scores(G, clients, directed)
            accounts["GARGAML_"+regime] = scores["GARGAML"]
            if directed: # the transpose-max variant of Eq. 14, for reference
                accounts["GARGAML_max_"+regime] = scores["GARGAML_max"]

            undirected_G = G.to_undirected() if directed else G
            n1, n2 = neighbourhood_sizes(undirected_G, clients)
            accounts["n1_"+regime], accounts["n2_"+regime] = n1, n2
            accounts["degree_"+regime] = n1

        # Degree is preserved by a view, so this is a check, not an assumption.
        differs = int((accounts["degree_full"] != accounts["degree_view"]).sum())
        print("  clients whose degree differs between regimes: "+str(differs)+
              " (expected 0 except for accounts held at two banks)")
        accounts["degree"] = accounts["degree_full"]

        frames.append(accounts.join(labels))

    accounts = pd.concat(frames)
    accounts_path = RESULTS_DIR+"/"+view+"_partial_observability_accounts.csv"
    accounts.to_csv(accounts_path)
    print("\n  accounts -> "+accounts_path)

    records = []
    for directed in DIRECTIONS:
        str_directed = "directed" if directed else "undirected"
        subset = accounts[accounts["direction"] == str_directed]
        records += evaluate_regimes(
            subset, labels, str_directed,
            dict(dataset=view, model=gargaml_key("base", directed), features="score"),
        )

    metrics = metrics_frame(records)
    metrics_path = RESULTS_DIR+"/"+view+"_partial_observability_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print("  metrics  -> "+metrics_path)
    print("  completed in "+format(timeit.default_timer()-start_time, ".1f")+" seconds")

    return accounts, metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for banks_spec in INSTITUTIONS:
        run_institution(banks_spec)
    print("\nAll institutions processed successfully.")


if __name__ == "__main__":
    echo_config(__file__, dataset=DATASET, institutions=INSTITUTIONS, n_cpu=n_cpu,
                results_dir=RESULTS_DIR)
    main()
