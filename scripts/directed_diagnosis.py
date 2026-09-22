"""
Directed-vs-undirected diagnosis (task 6) -- the experiment for R2-M2.

The reviewer asks why the undirected score wins when Section 3.3 motivates
the directed one from uni-directional flow, and names two hypotheses:
(a) benign bidirectional edges are over-penalised, (b) the level-assignment
rule (Eq. 11) is too strict. This script measures both on the synthetic
grid, where pattern membership is ground truth, and writes one row per node
plus a tidy metrics frame comparing five score variants on the same
neighbourhoods.

The mechanics live in ``src/methods/directed_diagnosis.py``; read that
module's docstring first -- it defines the variants and explains why the
reciprocal-edge count is exact rather than approximate.

What to look for in the output
------------------------------
``<dataset>_directed_diagnosis.csv`` (per node)
    Level census, reciprocal census and every score variant. The columns
    that carry the argument are ``d2_reverse_only`` (distance-2 nodes Eq.
    11 puts at level 2 where §3.3 puts them at level 0), ``d2_neither``
    (distance-2 nodes Eq. 11 cannot place at all, which it groups with the
    node itself) and ``reciprocal_penalised``.
``<dataset>_directed_diagnosis_summary.csv``
    The same, averaged within each ground-truth class.
``<dataset>_undirected_diagnosis_metrics.csv``
    AUC-ROC / AUC-PR / ranking metrics for every variant against every
    ground-truth column, through the shared evaluation module. This is the
    "does the gap close" table: compare ``gargaml_d`` against
    ``gargaml_d_flowsplit`` (hypothesis b) and ``gargaml_d_norecip``
    (hypothesis a), with ``gargaml_u`` as the target to close to.

Louvain
-------
Off by default. The diagnosis is about the level-assignment rule, and
running it on the reduced graph would mix Eq. 11's behaviour with the
edge-removal step's -- the same argument that made
scripts/partial_observability.py turn Louvain off on both sides. Set
``LOUVAIN = True`` to see the numbers the published pipeline would produce;
the ``directed`` column then reproduces ``results/<dataset>_GARGAML_
directed.csv`` exactly, which :func:`check_parity` verifies on a sample
either way.

Cost
----
Per node this builds three ego graphs and four adjacency matrices, so it is
several times the cost of one scoring pass. The 22 synthetic datasets at
n=100 run in seconds; the n=10,000 tier takes minutes per dataset. HI-Small
is feasible only sampled -- ``SAMPLE_NODES`` caps the node count and the
sample is seeded, and a sampled run is honest for the level and reciprocal
censuses but **not** for the ranking metrics, which are written only for a
complete pass.
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.graph_construction import construct_IBM_graph, construct_synthetic_graph
from src.methods.directed_diagnosis import (SCORE_VARIANTS, check_against_pipeline,
                                            diagnose_node)
from src.utils.evaluation import (SEED, evaluate_scores, metric_records,
                                  nan_metrics, write_metrics)
from src.utils.graph_processing import graph_community
from src.utils.runtime import (env_override, select_datasets, echo_config, as_list,
                              resolve_results_dir)

# See the module docstring: off by default, deliberately.
LOUVAIN = False

# Cap for a dataset too large to diagnose exhaustively. None = every node.
# A capped run still writes the per-node and summary files; it writes no
# metrics file, because ranking metrics over a sample are not the ranking
# metrics of the dataset.
SAMPLE_NODES = 20000

# How many nodes to check against the published pipeline before trusting a
# run's ``directed`` column.
PARITY_SAMPLE = 25

# The ground-truth columns of the synthetic label files.
SYNTHETIC_LABELS = ["laundering", "separate", "new_mules", "existing_mules"]

# Model keys for the tidy metrics frame. The published pair keeps its
# canonical naming.py keys; the variants are diagnosis-only and never reach
# a paper table under these names, which is also why this script writes with
# write_matrices=False (see write_metrics).
VARIANT_KEYS = {
    "directed":      "gargaml_d",
    "undirected":    "gargaml_u",
    "transpose":     "gargaml_d_transpose",
    "max_transpose": "gargaml_d_maxtranspose",
    "flow_split":    "gargaml_d_flowsplit",
    "no_reciprocal": "gargaml_d_norecip",
    "directed_prefix": "gargaml_d_prefix",
}


def synthetic_datasets(sizes=(100,)):
    """The synthetic grid, restricted to the given node counts.

    Mirrors ``construct_datasets`` in the measure scripts rather than
    importing it, because that one is hard-wired to all three size tiers
    and this script defaults to the cheap one.
    """
    names = []
    for n_nodes in sizes:
        for n_patterns in (3, 5):
            if n_patterns > 0.06 * n_nodes:
                continue
            for m_edges in (1, 2, 5):
                names.append(f"synthetic_Barabasi-Albert_{n_nodes}_{m_edges}_0_{n_patterns}")
            for p_edges in (0.001, 0.01):
                names.append(f"synthetic_Erdos-Renyi_{n_nodes}_0_{p_edges}_{n_patterns}")
            for m_edges in (1, 2, 5):
                for p_edges in (0.001, 0.01):
                    names.append(f"synthetic_Watts-Strogatz_{n_nodes}_{m_edges}_{p_edges}_{n_patterns}")
    return names


DATASETS = synthetic_datasets()

# Slurm overrides; the constants above remain the documented defaults.
# A capped run (SAMPLE_NODES) deliberately writes no metrics file.
DATASETS = select_datasets(DATASETS)
SAMPLE_NODES = env_override("sample_nodes", SAMPLE_NODES,
                            lambda r: None if r.lower() in ("none", "all") else int(r))
RESULTS_DIR = resolve_results_dir()
# Widen when the budget allows. HI-Small is only meaningful sampled -- see
# SAMPLE_NODES and the caveat in the module docstring.
# DATASETS = synthetic_datasets((100, 10000))
# DATASETS = synthetic_datasets() + ["HI-Small"]


def build_graph(dataset):
    """The directed graph, its undirected view and its reverse."""
    if dataset.startswith("synthetic"):
        G = construct_synthetic_graph("data/edge_data_"+dataset+".csv", directed=True)
    else:
        G = construct_IBM_graph("data/"+dataset+"_Trans.csv", directed=True)

    if LOUVAIN:
        G = graph_community(G)

    return G, G.to_undirected(), G.reverse(copy=True)


def check_parity(nodes, G, G_und, G_rev, rng):
    """Verify the diagnosis reproduces the pipeline on a sample of nodes.

    The whole table is only readable if its ``directed`` column is the
    number the pipeline writes, so this is checked rather than assumed --
    and checked per dataset, because the level assignment depends on the
    graph, not just on the code.
    """
    sample = rng.choice(len(nodes), size=min(PARITY_SAMPLE, len(nodes)),
                        replace=False)
    for i in sample:
        check_against_pipeline(nodes[i], G, G_und, G_rev)
    return len(sample)


def structural_roles(G, labelled):
    """Approximate source / mule / target role for each labelled node.

    Derived from the directed edges **between labelled nodes**: a source
    only pays out, a target only receives, a mule does both. Exact for the
    ``separate`` patterns, which are disconnected components; approximate
    for ``new_mules`` and ``existing_mules``, where a participant can also
    border a labelled node from another pattern. Reported because it is
    what makes the diagnosis legible -- Eq. 14 is oriented correctly for
    sources and backwards for targets -- not as a published quantity.
    """
    labelled = set(labelled)
    roles = {}
    for n in labelled:
        out = sum(1 for s in G.successors(n) if s in labelled)
        inc = sum(1 for p in G.predecessors(n) if p in labelled)
        if out and inc:
            roles[n] = "mule"
        elif out:
            roles[n] = "source"
        elif inc:
            roles[n] = "target"
        else:
            roles[n] = "isolated"
    return roles


def load_labels(dataset, nodes):
    """Ground-truth membership for ``nodes``, or ``None`` if unavailable.

    Synthetic label files are indexed by node id; a node with no row is
    not part of any pattern, which is a 0 rather than a missing value.
    """
    if not dataset.startswith("synthetic"):
        return None
    path = "data/label_data_"+dataset+".csv"
    if not os.path.exists(path):
        return None
    labels = pd.read_csv(path)[SYNTHETIC_LABELS].astype(int)
    return labels.reindex(nodes).fillna(0).astype(int)


def diagnose_dataset(dataset):
    """One dataset end to end: per-node rows, summary, and the metrics frame."""
    print("\n=== "+dataset+" ===")
    G, G_und, G_rev = build_graph(dataset)
    nodes = list(G.nodes)
    print("  "+str(len(nodes))+" nodes, "+str(G.number_of_edges())+" directed edges"
          +(" (Louvain-reduced)" if LOUVAIN else " (no Louvain -- see module docstring)"))

    rng = np.random.default_rng(SEED)
    print("  parity with the published pipeline verified on "
          +str(check_parity(nodes, G, G_und, G_rev, rng))+" sampled nodes")

    complete = SAMPLE_NODES is None or len(nodes) <= SAMPLE_NODES
    if not complete:
        idx = rng.choice(len(nodes), size=SAMPLE_NODES, replace=False)
        nodes = [nodes[i] for i in sorted(idx)]
        print("  SAMPLED to "+str(len(nodes))+" nodes: level and reciprocal "
              "censuses are valid, ranking metrics are not and will be skipped")

    rows = [diagnose_node(n, G, G_und, G_rev) for n in tqdm(nodes, desc="  nodes")]
    df = pd.DataFrame(rows).set_index("node")

    labels = load_labels(dataset, df.index)
    if labels is not None:
        df = df.join(labels)
        roles = structural_roles(G, labels.index[labels["laundering"] == 1])
        df["role"] = pd.Series(roles).reindex(df.index).fillna("none")

    path = RESULTS_DIR+"/"+dataset+"_directed_diagnosis.csv"
    df.to_csv(path)
    print("  per-node diagnosis -> "+path)

    write_summary(df, dataset, labels is not None)

    if labels is not None and complete:
        write_variant_metrics(df, dataset, labels)

    return df


def write_summary(df, dataset, has_labels):
    """Group means, by ground-truth class and by structural role."""
    columns = [c for c in df.columns
               if c.startswith(("n_level", "d2_", "gain_", "gap_", "reciprocal_",
                                "empty_level", "prefix_"))
               or c in SCORE_VARIANTS
               or c in ("undirected", "directed_prefix", "n_edges", "n_reciprocal")]

    frames = [df[columns].mean().rename("all").to_frame().T.assign(group="all")]
    if has_labels:
        for label in SYNTHETIC_LABELS:
            positives = df[df[label] == 1]
            if len(positives):
                frames.append(positives[columns].mean().to_frame().T.assign(group=label))
        for role, group in df[df["role"] != "none"].groupby("role"):
            frames.append(group[columns].mean().to_frame().T.assign(group="role:"+role))

    summary = pd.concat(frames, ignore_index=True)
    summary.insert(0, "group", summary.pop("group"))
    summary.insert(0, "dataset", dataset)

    path = RESULTS_DIR+"/"+dataset+"_directed_diagnosis_summary.csv"
    summary.to_csv(path, index=False)
    print("  summary -> "+path)
    return summary


def write_variant_metrics(df, dataset, labels):
    """Ranking + AUC metrics for every variant, against every label column.

    Goes through the shared evaluation module (task 2), so the variants are
    measured with exactly the metrics the rest of the revision reports.
    ``write_matrices=False``: these are diagnosis-only model keys with no
    historical matrix format behind them, and inventing one would create
    files no notebook reads.
    """
    records = []
    for target in SYNTHETIC_LABELS:
        y_true = labels[target].values
        for variant, model_key in VARIANT_KEYS.items():
            context = dict(dataset=dataset, direction="directed", model=model_key,
                           features="score", cutoff=np.nan, target=target, seed=SEED)
            try:
                # No natural 0/1 prediction for a raw score, so precision and
                # f1 come back NaN -- same convention as distribution_scores.py.
                metrics = evaluate_scores(y_true, df[variant].values)
                status = "ok"
            except Exception as exc:  # a label column with no positives here
                metrics, status = nan_metrics(), "skipped: "+str(exc)
            records += metric_records(metrics, status=status, n_test=len(y_true),
                                      n_pos=int(y_true.sum()), **context)

    return write_metrics(records, dataset, "directed", suffix="_diagnosis",
                         results_dir=RESULTS_DIR, write_matrices=False)


def main():
    summaries, metrics = [], []
    for dataset in DATASETS:
        if not os.path.exists("data/edge_data_"+dataset+".csv") and \
           not os.path.exists("data/"+dataset+"_Trans.csv"):
            print("\n### "+dataset+" -- SKIPPED: no edge data on disk ###")
            continue
        diagnose_dataset(dataset)
        summaries.append(RESULTS_DIR+"/"+dataset+"_directed_diagnosis_summary.csv")
        path = RESULTS_DIR+"/"+dataset+"_directed_diagnosis_metrics.csv"
        if os.path.exists(path):
            metrics.append(path)

    # A sharded/array submission (one dataset per task, via GARGAML_DATASET /
    # GARGAML_DATASET_INDEX / SLURM_ARRAY_TASK_ID) only ever sees its own
    # dataset in DATASETS, so the pooled write below would silently clobber
    # the fixed-name pooled files with just that one dataset's rows,
    # discarding every other task's. Skip the pooled write in that case; the
    # per-dataset files above are written normally either way.
    sharded = any(os.environ.get(k) for k in
                  ("GARGAML_DATASET", "GARGAML_DATASET_INDEX", "SLURM_ARRAY_TASK_ID"))
    if sharded:
        print("skipping pooled directed_diagnosis_summary/metrics.csv: this looks "
              "like a sharded run (GARGAML_DATASET/_INDEX or SLURM_ARRAY_TASK_ID is "
              "set) -- rerun over the full DATASETS list to regenerate the pooled files")
        return

    # Pool the grid, which is where the argument actually lives: one
    # 100-node dataset is too small to carry it.
    if summaries:
        pooled = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
        pooled.to_csv(RESULTS_DIR+"/directed_diagnosis_summary.csv", index=False)
        print("\npooled summary over "+str(len(summaries))+" datasets -> "
              +RESULTS_DIR+"/directed_diagnosis_summary.csv")
    if metrics:
        pooled = pd.concat([pd.read_csv(p) for p in metrics], ignore_index=True)
        pooled.to_csv(RESULTS_DIR+"/directed_diagnosis_metrics.csv", index=False)
        print("pooled variant metrics -> "+RESULTS_DIR+"/directed_diagnosis_metrics.csv")


if __name__ == "__main__":
    echo_config(__file__, datasets=DATASETS, sample_nodes=SAMPLE_NODES, louvain=LOUVAIN)
    main()
