"""
Block-only ablation of GARG-AML (task 3).

Runs the shared tree/boosting path of ``scripts/gargaml_tree.py`` on the
``blocks`` feature config: only the per-node *block densities and block
sizes* produced by the adjacency-matrix block analysis. No degree
features, no neighbour aggregations, no aggregated GARG-AML score.

It is one of the three ablations the reviewer asked for:

* topology-only (``topology``) -> only own + neighbour degree stats,
                                  no GARG-AML at all
* block-only (``blocks``, here) -> only per-node block densities + sizes,
                                   before aggregation
* the published model (``full``) -> aggregated score + neighbour score
                                    stats + degree stats
* (``all``)                      -> literally all four groups

Together they answer "how much of the lift comes from the GARG-AML block
layout, and how much from the neighbour-degree summary". The column
groups themselves live in ``src/utils/features.py``; the train/evaluate
loop lives in ``scripts/gargaml_tree.py``. This file exists so the
ablation can be re-run on its own -- ``gargaml_tree.py`` runs all four
configs off one data preparation, which is the cheaper way to get the
full grid.

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

from scripts.gargaml_tree import data_preparation, run_config
from src.utils.features import all_feature_columns

CONFIG = "blocks"


def main():
    # Iterate over both directions so one invocation gives the full
    # ablation grid (undirected + directed x tree + boosting).
    dataset = "HI-Small"
    score_type = "weighted_average"

    for directed in [False, True]:
        # The block columns come straight from the measures CSV, so this
        # preparation needs neither the graph nor Louvain.
        laundering_combined = data_preparation(
            dataset, all_feature_columns([CONFIG], directed), directed, score_type
        )
        run_config(laundering_combined, dataset, directed, CONFIG)


if __name__ == "__main__":
    main()
