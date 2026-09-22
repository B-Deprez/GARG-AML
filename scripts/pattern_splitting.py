"""
How many laundering patterns the Louvain step destroys (task 4, R2-M3).

The sweep in the measure scripts answers "how much does the pre-processing
discard"; this answers "what does it discard". For every laundering attempt
in the IBM patterns file, at every resolution in the sweep, it records
whether the attempt's accounts stayed in one community, how many of its own
edges survived, and -- the number that matters -- whether its two-hop
source/mule/target structure survived at all. Once that is gone GARG-AML
cannot see the pattern, whatever the score threshold.

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

Read it beside ``results/louvain_severance.csv``: the pair is the argument.
Edges severed says what the step costs, patterns destroyed says what it
costs *us*.

Cost
----
The graph is built **once** and only Louvain is re-run per resolution, which
is what makes the whole sweep affordable -- graph construction from the
475 MB transactions file dominates a single-resolution run. Expect a few
minutes for HI-Small, most of it before the first resolution.

Ground truth
------------
Exact, not inferred: ``data/<dataset>_Patterns.txt`` delimits each attempt
with its own transaction list, so attempt membership is read rather than
reconstructed. HI-Small holds 370 attempts, 95 of them the GATHER-SCATTER /
SCATTER-GATHER shapes GARG-AML targets.
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

# Datasets to diagnose. LI-Large is listed but is the multi-hour job -- its
# graph alone is 176M edges.
DATASETS = ["HI-Small"]
# DATASETS = ["HI-Small", "LI-Large"]

# The sweep, matching the arms in the measure scripts. ``None`` is the
# no-Louvain control: nothing is partitioned, so every attempt survives
# intact by construction. It is kept in the list rather than assumed,
# because a control that is computed and comes out at 100 % is evidence that
# the measurement is doing what it claims.
RESOLUTIONS = [None, 1, 5, DEFAULT_RESOLUTION, 20, 50]

# Slurm overrides; the constants above remain the documented defaults.
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
            # No partition at all: one community for everything, so nothing
            # is severed. Built explicitly rather than special-cased inside
            # the analysis, so the control goes down the same code path.
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

    # A sharded/array submission (one dataset per task, via GARGAML_DATASET /
    # GARGAML_DATASET_INDEX / SLURM_ARRAY_TASK_ID) only ever sees its own
    # dataset in DATASETS, so the pooled write below would silently clobber
    # the fixed-name pooled file with just that one dataset's rows,
    # discarding every other task's. Skip it in that case; the per-dataset
    # files above are written normally either way.
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
