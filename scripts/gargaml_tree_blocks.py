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

import pandas as pd
from sklearn import ensemble, tree

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.utils.evaluation import (
    SEED,
    evaluate_model,
    holdout_split,
    metric_records,
    nan_metrics,
    write_metrics,
)
from src.utils.naming import gargaml_key


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


def data_split(combined, feature_cols, target, cutoff, seed=SEED):
    X = combined[feature_cols]
    y = (combined[target] > cutoff).astype(int).values
    X_train, X_test, y_train, y_test = holdout_split(X, y, seed=seed)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_direction(dataset, directed, seed=SEED):
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

    models = [
        (gargaml_key("tree", directed), gargaml_tree_blocks),
        (gargaml_key("boost", directed), gargaml_boosting_blocks),
    ]

    records = []

    for cutoff in cut_offs:
        for target in columns:
            print(f"  cutoff={cutoff}  target={target}")

            context = dict(
                dataset = dataset,
                direction = str_directed,
                features = "blocks", # group (b) only: block densities + sizes, no degree/score stats
                cutoff = cutoff,
                target = target,
                seed = seed,
            )

            try:
                X_train, X_test, y_train, y_test = data_split(
                    combined, feature_cols, target, cutoff, seed=seed
                )
            except Exception as exc: # Too few labels to even split: no models for this cell
                print(f"    no split: {exc!r}")
                records += metric_records({"imbalance": 0.0}, status = "skipped: "+str(exc), model = "", **context)
                for model_key, _ in models:
                    records += metric_records(nan_metrics(), status = "skipped: "+str(exc), model = model_key, **context)
                continue

            n_test = len(y_test)
            n_pos = int(sum(y_test))

            records += metric_records(
                {"imbalance": y_train.sum() / len(y_train)},
                n_test = n_test, n_pos = n_pos, model = "", **context
                )

            for model_key, fit_model in models:
                try: # If too few labels, the model will not work. The cell is reported as NaN, with the reason
                    clf = fit_model(X_train, y_train)
                    metrics = evaluate_model(clf, X_test, y_test)
                    status = "ok"
                except Exception as exc:
                    print(f"    {model_key} skipped: {exc!r}")
                    metrics = nan_metrics()
                    status = "skipped: "+str(exc)

                records += metric_records(
                    metrics, status = status, n_test = n_test, n_pos = n_pos,
                    model = model_key, **context
                    )

    write_metrics(records, dataset, str_directed, suffix="_blocks")


def main():
    # Iterate over both directions so one invocation gives the full
    # ablation grid (undirected + directed x tree + boosting).
    dataset = "HI-Small"
    for directed in [False, True]:
        run_one_direction(dataset, directed)


if __name__ == "__main__":
    main()
