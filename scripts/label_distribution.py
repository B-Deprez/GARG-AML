"""
Regenerate Tables 7-8's label-percentage matrices with the 0.0 cut-off.

``CUT_OFFS`` (``src/utils/evaluation.py``) added the 0.0 cut-off during the
revision, but the four ``<dataset>_imbalance_<direction>_combined.csv`` files
on disk predate it and only carry the published 0.1/0.2/0.3/0.5/0.9 rows (see
CLAUDE.md). Those files are exactly Tables 7-8: ``notebooks/DistributionScores.ipynb``
(cells 29-32) reads one, scales it by 100 and prints ``metrics_p.T.to_latex()``.

``imbalance`` is model- and direction-independent -- it is the population
proportion labelled 1 at a (cutoff, target) cell, computed straight from
``src/data/pattern_construction.py`` -- so this bypasses
``gargaml_tree.py``'s ``data_preparation()`` (GARG-AML scores, Louvain
reduction, every tree/boosting fit) entirely via
``src/utils/evaluation.py::label_imbalance_records``. That keeps this cheap
even for LI-Large; the model tables (10-11) are a separate, more expensive
rerun (see CLAUDE.md's task 2 note on the stale tree grid).

Run from the repository root::

    python scripts/label_distribution.py

Writes the same four files ``notebooks/DistributionScores.ipynb`` already
reads, so that notebook needs no changes -- just re-run its cells.
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from src.utils.evaluation import (label_imbalance_records, metrics_frame,
                                  write_metric_matrices)
from src.utils.runtime import resolve_results_dir, select_datasets

RESULTS_DIR = resolve_results_dir()

# A missing dataset is skipped with a message. Override with GARGAML_DATASET
# (or GARGAML_DATASETS) to run one at a time -- LI-Large's read alone takes a
# while, so it's worth separating from HI-Small's near-instant run.
DATASETS = select_datasets(["HI-Small", "LI-Large"])

if __name__ == "__main__":
    for dataset in DATASETS:
        path_trans = "data/" + dataset + "_Trans.csv"
        if not os.path.exists(path_trans):
            print(dataset + ": no " + path_trans + " on disk, skipping.")
            continue

        print("==== " + dataset + " ====")
        records = label_imbalance_records(dataset)
        long_df = metrics_frame(records)

        # Written under both direction tokens: the value is the same either
        # way, and the notebook picks the file by direction in its filename.
        for direction in ["undirected", "directed"]:
            written = write_metric_matrices(long_df, dataset, direction,
                                            results_dir=RESULTS_DIR)
            print("  " + direction + ": " + str(written))
