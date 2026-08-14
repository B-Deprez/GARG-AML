"""
Block-only ablation of GARG-AML.

Trains the same decision-tree and gradient-boosting classifiers as
``scripts/gargaml_tree.py``, but on a restricted feature set consisting
only of the per-node *block densities and block sizes* produced by the
adjacency-matrix block analysis. No degree features, no neighbour
aggregations, no aggregated GARG-AML score.

This is the natural complement to the topology-only baseline requested
by the reviewer:

* topology-only baseline   -> only degree neighbour stats
                              (`degree, degree_min, ..., degree_std`),
                              no GARG-AML at all
* block-only ablation (here) -> only per-node block densities + sizes,
                                no degree at all
* full GARG-AML model       -> both feature families combined
                              (scripts/gargaml_tree.py)

Together the three answer "how much of the lift comes from the GARG-AML
block layout vs. the neighbour-degree summary".

Input CSVs (already produced by gargaml_undirected.py / gargaml_directed.py):

* ``results/<dataset>_GARGAML_undirected.csv`` with columns
  ``node, measure_1, measure_2, measure_3, size_1, size_2, size_3``.
* ``results/<dataset>_GARGAML_directed.csv`` with columns
  ``node, measure_00, ..., measure_22, size_00, ..., size_22``.

Output CSVs (mirror gargaml_tree.py's naming with a ``_blocks`` suffix):

* ``results/<dataset>_<metric>_<model>_<direction>_blocks_combined.csv``
* ``results/<dataset>_imbalance_<direction>_blocks_combined.csv``
* ``results/<dataset>_<direction>_blocks_feature_schema.csv`` documenting
  the exact features fed to the models (for the appendix).
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from pickle import dump

import numpy as np
import pandas as pd
from sklearn import ensemble, tree
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

UNDIRECTED_FEATURES = [
    "measure_1", "measure_2", "measure_3",
    "size_1", "size_2", "size_3",
]

DIRECTED_FEATURES = (
    [f"measure_{i}{j}" for i in range(3) for j in range(3)]
    + [f"size_{i}{j}" for i in range(3) for j in range(3)]
)


def block_features(directed):
    return DIRECTED_FEATURES if directed else UNDIRECTED_FEATURES


# ---------------------------------------------------------------------------
# Model fit + evaluation (kept identical to scripts/gargaml_tree.py)
# ---------------------------------------------------------------------------

def gargaml_tree_blocks(X, y, save=False, save_path="results/model_tree_blocks.pkl"):
    clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
    clf = clf.fit(X, y)
    if save:
        with open(save_path, "wb") as f:
            dump(clf, f, protocol=5)
    return clf


def gargaml_boosting_blocks(X, y, save=False, save_path="results/model_boosting_blocks.pkl"):
    clf = ensemble.GradientBoostingClassifier(min_samples_leaf=10, random_state=1997)
    clf = clf.fit(X, y)
    if save:
        with open(save_path, "wb") as f:
            dump(clf, f, protocol=5)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_pred)
    return precision, f1, auc_roc, auc_pr


# ---------------------------------------------------------------------------
# Data preparation: per-account block features joined with AML labels
# ---------------------------------------------------------------------------

def data_preparation_blocks(dataset, directed):
    str_directed = "directed" if directed else "undirected"
    feature_cols = block_features(directed)

    measures_path = f"results/{dataset}_GARGAML_{str_directed}.csv"
    measures_df = pd.read_csv(measures_path)
    missing = [c for c in feature_cols if c not in measures_df.columns]
    if missing:
        raise RuntimeError(
            f"{measures_path} is missing block columns: {missing}. "
            f"Did you run scripts/gargaml_{str_directed}.py first?"
        )
    measures_df = measures_df.set_index("node")[feature_cols]

    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans=f"data/{dataset}_Trans.csv",
        path_patterns=f"data/{dataset}_Patterns.txt",
    )
    laundering_combined, _, _ = summarise_ML_labels(
        transactions_df_extended, pattern_columns
    )

    # Inner-join on account: drop accounts that don't have block features
    # (e.g. dropped by Louvain edge filtering) and accounts without labels.
    combined = laundering_combined.join(measures_df, how="inner")
    return combined, feature_cols


def data_split(combined, feature_cols, target, cutoff):
    X = combined[feature_cols]
    y = (combined[target] > cutoff).astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1997, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_direction(dataset, directed):
    str_directed = "directed" if directed else "undirected"
    cut_offs = [0.1, 0.2, 0.3, 0.5, 0.9]
    columns = [
        "Is Laundering",
        "FAN-OUT", "FAN-IN",
        "GATHER-SCATTER", "SCATTER-GATHER",
        "CYCLE", "RANDOM", "BIPARTITE", "STACK",
    ]

    print(f"\n=== Block-only ablation: {dataset} ({str_directed}) ===")
    combined, feature_cols = data_preparation_blocks(dataset, directed)
    print(f"  features ({len(feature_cols)}): {feature_cols}")
    print(f"  joined dataframe shape: {combined.shape}")

    # Persist the feature schema for the appendix.
    schema_path = f"results/{dataset}_{str_directed}_blocks_feature_schema.csv"
    pd.DataFrame({
        "feature": feature_cols,
        "kind": ["density" if c.startswith("measure_") else "size" for c in feature_cols],
    }).to_csv(schema_path, index=False)
    print(f"  feature schema -> {schema_path}")

    n, m = len(cut_offs), len(columns)
    metric_matrices = {
        "precision_tree":     np.zeros((n, m)),
        "precision_boosting": np.zeros((n, m)),
        "f1_tree":            np.zeros((n, m)),
        "f1_boosting":        np.zeros((n, m)),
        "AUC_ROC_tree":       np.zeros((n, m)),
        "AUC_ROC_boosting":   np.zeros((n, m)),
        "AUC_PR_tree":        np.zeros((n, m)),
        "AUC_PR_boosting":    np.zeros((n, m)),
    }
    imbalance = np.zeros((n, m))

    for i, cutoff in enumerate(cut_offs):
        for j, target in enumerate(columns):
            print(f"  cutoff={cutoff}  target={target}")
            try:
                X_train, X_test, y_train, y_test = data_split(
                    combined, feature_cols, target, cutoff
                )
                imbalance[i, j] = y_train.sum() / len(y_train)

                tree_clf = gargaml_tree_blocks(X_train, y_train)
                p_t, f_t, r_t, pr_t = evaluate_model(tree_clf, X_test, y_test)

                boost_clf = gargaml_boosting_blocks(X_train, y_train)
                p_b, f_b, r_b, pr_b = evaluate_model(boost_clf, X_test, y_test)

                metric_matrices["precision_tree"][i, j] = p_t
                metric_matrices["f1_tree"][i, j] = f_t
                metric_matrices["AUC_ROC_tree"][i, j] = r_t
                metric_matrices["AUC_PR_tree"][i, j] = pr_t

                metric_matrices["precision_boosting"][i, j] = p_b
                metric_matrices["f1_boosting"][i, j] = f_b
                metric_matrices["AUC_ROC_boosting"][i, j] = r_b
                metric_matrices["AUC_PR_boosting"][i, j] = pr_b
            except Exception as exc:
                # Same try/except pattern as gargaml_tree.py: too few positives
                # in this (cutoff, pattern) cell -> NaN entry, not an error.
                print(f"    skipped: {exc!r}")
                for key in metric_matrices:
                    metric_matrices[key][i, j] = np.nan

    for name, mat in metric_matrices.items():
        out = pd.DataFrame(mat, columns=columns, index=cut_offs)
        out.to_csv(f"results/{dataset}_{name}_{str_directed}_blocks_combined.csv")
    pd.DataFrame(imbalance, columns=columns, index=cut_offs).to_csv(
        f"results/{dataset}_imbalance_{str_directed}_blocks_combined.csv"
    )
    print(f"  wrote {len(metric_matrices) + 1} result CSVs with suffix _blocks_combined")


def main():
    # Iterate over both directions so one invocation gives the full
    # ablation grid (undirected + directed x tree + boosting).
    dataset = "HI-Small"
    for directed in [False, True]:
        run_one_direction(dataset, directed)


if __name__ == "__main__":
    main()
