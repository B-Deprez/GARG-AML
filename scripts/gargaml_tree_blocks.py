"""
Block-only ablation of GARG-AML.

Runs the shared tree/boosting path of ``scripts/gargaml_tree.py`` on the
``blocks`` feature config: only the per-node *block densities and block
sizes* produced by the adjacency-matrix block analysis. No degree
features, no neighbour aggregations, no aggregated GARG-AML score, which
isolates how much of the lift comes from the block layout itself.

The column groups live in ``src/utils/features.py``; the train/evaluate
loop lives in ``scripts/gargaml_tree.py``. This file exists so the
ablation can be re-run on its own -- ``gargaml_tree.py`` runs all four
feature configs off one data preparation, which is the cheaper way to
get the full grid.

Input CSVs (already produced by gargaml_undirected.py / gargaml_directed.py):

* ``results/<dataset>_GARGAML_undirected.csv`` with columns
  ``node, measure_1, measure_2, measure_3, size_1, size_2, size_3``.
* ``results/<dataset>_GARGAML_directed.csv`` with columns
  ``node, measure_00, ..., measure_22, size_00, ..., size_22``.

Output CSVs (gargaml_tree.py's naming with a ``_blocks`` suffix):

* ``results/<dataset>_<metric>_<model>_<direction>_blocks_combined.csv``
* ``results/<dataset>_imbalance_<direction>_blocks_combined.csv``
* ``results/<dataset>_<direction>_blocks_metrics.csv`` -- the tidy frame
* ``results/<dataset>_<direction>_blocks_feature_schema.csv`` documenting
  the exact features fed to the models (for the appendix).
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd

from scripts.gargaml_tree import (N_FOLDS, RESULTS_DIR, data_preparation,
                                  dataset_settings, run_config,
                                  write_fold_partition)
from src.utils.evaluation import folds_path
from src.utils.features import all_feature_columns
from src.utils.runtime import echo_config, select_datasets

CONFIG = "blocks"

# HI-Small unless GARGAML_DATASET (slurm/tree_blocks.slurm's argument) names
# another, resolved the same way gargaml_tree.py resolves its own list.
DATASETS = select_datasets(["HI-Small"])


def check_fold_partition(laundering_combined, dataset):
    """Keep the blocks fits on the partition the other models are scored on.

    run_config re-derives its folds from (labels, N_FOLDS, seed), so it
    trains on whatever N_FOLDS says, while the other configs, GraphSAGE and
    the base score all read the partition gargaml_tree.py persisted. A run
    under a different GARGAML_N_FOLDS would put the blocks results on folds
    no other model used, so that is refused; a missing partition is written,
    exactly as gargaml_tree.py would write it.
    """
    if N_FOLDS < 2:
        return
    path = folds_path(dataset, results_dir=RESULTS_DIR)
    if not os.path.exists(path):
        cut_offs, targets = dataset_settings(dataset)
        path = write_fold_partition(laundering_combined, dataset, cut_offs, targets, N_FOLDS)
        print("  folds -> "+path)
        return
    n_saved = pd.read_csv(path, usecols=["fold"])["fold"].nunique()
    if n_saved != N_FOLDS:
        raise ValueError(
            path+" holds a "+str(n_saved)+"-fold partition but N_FOLDS = "
            +str(N_FOLDS)+", so the blocks results would not be paired with the "
            "other models'. Set GARGAML_N_FOLDS="+str(n_saved)+", or rerun "
            "gargaml_tree.py with N_FOLDS = "+str(N_FOLDS)+" first.")
    print("  folds: "+path+" ("+str(n_saved)+" folds, matches N_FOLDS)")


def main():
    # Both directions in one invocation gives the full ablation grid
    # (undirected + directed x tree + boosting).
    score_type = "weighted_average"

    for dataset in DATASETS:
        for directed in [False, True]:
            # Block columns come straight from the measures CSV, so this needs
            # neither the graph nor Louvain.
            laundering_combined = data_preparation(
                dataset, all_feature_columns([CONFIG], directed), directed, score_type
            )
            # The partition depends on the labels only, not the direction.
            if not directed:
                check_fold_partition(laundering_combined, dataset)
            run_config(laundering_combined, dataset, directed, CONFIG)


if __name__ == "__main__":
    echo_config(__file__, datasets=DATASETS, config=CONFIG, n_folds=N_FOLDS,
                results_dir=RESULTS_DIR)
    main()
