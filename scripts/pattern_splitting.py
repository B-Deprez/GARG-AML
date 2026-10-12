"""
How many laundering patterns the Louvain reduction splits across communities.

For every laundering attempt in the IBM patterns file, at every resolution
in the sweep, this records whether the attempt's accounts stayed in one
community, how many of its own edges survived, and whether its two-hop
source/mule/target structure survived at all. Once that structure is gone
GARG-AML cannot see the pattern, whatever the score threshold.

Run from the repository root::

    python scripts/pattern_splitting.py

Outputs
-------
``results/<dataset>_pattern_splitting.csv``
    One row per (attempt, resolution): communities spanned, edge survival,
    2-path survival, and the ``detectable_before`` / ``detectable_after``
    pair.
``results/<dataset>_pattern_splitting_summary.csv``
    The same by pattern type, with ``pct_destroyed`` -- the share of
    two-hop attempts that no longer have a 2-path.
``results/pattern_splitting_summary.csv``
    Every dataset pooled, for the table.

Companion to ``results/louvain_severance.csv``, which records how many edges
the same reduction severs.

The graph is built once and only Louvain is re-run per resolution; graph
construction from the transactions file dominates the runtime. Attempt
membership is exact rather than inferred: ``data/<dataset>_Patterns.txt``
delimits each attempt with its own transaction list.
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd

from src.data.bank_views import parse_view, patterns_path, trans_path
from src.data.graph_construction import construct_IBM_graph
from src.data.pattern_construction import pattern_instances
from src.utils.graph_processing import DEFAULT_RESOLUTION, community_map
from src.utils.pattern_splitting import analyse, summarise
from src.utils.runtime import (env_override, select_datasets, echo_config, as_list,
                              resolve_results_dir)

# Datasets to diagnose; LI-Large is the expensive run.
DATASETS = ["HI-Small"]
# DATASETS = ["HI-Small", "LI-Large"]

# The sweep, matching the arms in the measure scripts. ``None`` is the
# no-Louvain control: nothing is partitioned, so every attempt survives
# intact. It is computed rather than assumed, so the control goes through
# the same measurement as the rest.
RESOLUTIONS = [None, 1, 5, DEFAULT_RESOLUTION, 20, 50]

DATASETS = select_datasets(DATASETS)
RESULTS_DIR = resolve_results_dir()


def diagnose_dataset(dataset):
    """Every resolution for one dataset, off a single graph build."""
    print("\n=== "+dataset+" ===")

    base, banks = parse_view(dataset)
    G = construct_IBM_graph(path=trans_path(dataset), directed=True, banks=banks)
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    instances = pattern_instances(patterns_path(dataset))
    print(f"  ground truth: {instances.instance.nunique()} laundering attempts, "
          f"{len(instances)} transactions")

    nodes = set(G.nodes)
    frames = []
    for resolution in RESOLUTIONS:
        if resolution is None:
            # The no-Louvain control: one community for everything.
            communities = dict.fromkeys(nodes, 0)
        else:
            communities = community_map(G, resolution=resolution)

        frame = analyse(instances, nodes, communities, dataset, resolution)
        frames.append(frame)

        had = frame[frame["detectable_before"]]
        destroyed = (~had["detectable_after"]).mean() * 100 if len(had) else float("nan")
        print(f"  resolution={'off' if resolution is None else resolution:<4}"
              f"  split {100 * frame['split'].mean():5.1f}%"
              f"  edges kept {100 * frame['edge_survival'].mean():5.1f}%"
              f"  2-paths kept {100 * had['path_survival'].mean():5.1f}%"
              f"  DESTROYED {destroyed:5.1f}%")

    frame = pd.concat(frames, ignore_index=True)
    path = RESULTS_DIR+"/"+dataset+"_pattern_splitting.csv"
    frame.to_csv(path, index=False)
    print("  per-attempt -> "+path)

    summary = summarise(frame)
    path = RESULTS_DIR+"/"+dataset+"_pattern_splitting_summary.csv"
    summary.to_csv(path, index=False)
    print("  summary     -> "+path)
    return summary


def main():
    summaries = []
    for dataset in DATASETS:
        if not os.path.exists(patterns_path(dataset)):
            print("\n### "+dataset+" -- SKIPPED: "+patterns_path(dataset)+" not found ###")
            continue
        summaries.append(diagnose_dataset(dataset))

    if not summaries:
        return

    # A sharded/array submission (one dataset per task) only ever sees its own
    # dataset in DATASETS, so the pooled write below would clobber the
    # fixed-name pooled file with that one dataset's rows. Skip it in that
    # case; the per-dataset files above are written either way.
    sharded = any(os.environ.get(k) for k in
                  ("GARGAML_DATASET", "GARGAML_DATASET_INDEX", "SLURM_ARRAY_TASK_ID"))
    if sharded:
        print("skipping pooled pattern_splitting_summary.csv: this looks like a "
              "sharded run (GARGAML_DATASET/_INDEX or SLURM_ARRAY_TASK_ID is set) "
              "-- rerun over the full DATASETS list to regenerate the pooled file")
        return

    pooled = pd.concat(summaries, ignore_index=True)
    pooled.to_csv(RESULTS_DIR+"/pattern_splitting_summary.csv", index=False)
    print("\npooled summary -> "+RESULTS_DIR+"/pattern_splitting_summary.csv")
    print("\nGARG-AML's own targets:")
    targets = pooled[pooled.pattern_type.isin(["GATHER-SCATTER", "SCATTER-GATHER"])]
    print(targets[["dataset", "resolution", "pattern_type", "attempts",
                   "pct_split", "mean_path_survival", "pct_destroyed"]]
          .to_string(index=False))


if __name__ == "__main__":
    echo_config(__file__, datasets=DATASETS, resolutions=RESOLUTIONS)
    main()
