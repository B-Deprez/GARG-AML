"""
GraphSAGE baseline (task 1) -- minimal, tutorial-style first pass.

Deliberately narrow scope, per the task's own sequencing note ("get one
HI-Small run working end to end before queuing LI-Large") and an explicit
request to keep this as close to a plain PyTorch Geometric tutorial as
possible rather than building the full GARG-AML_code_changes.md task-1 spec
at once:

  * one feature config (topology-only: [degree, log-degree] on the raw,
    undirected, parallel-collapsed transaction graph -- never GARG-AML scores
    or block measures)
  * a small CUT_OFFS/TARGET_COLUMNS sweep, extensible the same way
    gargaml_tree.py's is
  * HI-Small only

It *is* wired into task 7's persisted results/<dataset>_folds.csv, so results
are fold-by-fold comparable to the tree/boosting numbers from day one -- that
was an explicit choice, not a simplification.

Deliberately deferred to a later pass, not silently dropped:
  * the second ("generous") feature config with amount/timing attributes
  * the full 9-cutoff sweep and LI-Large
  * early stopping on a validation split, timing/peak-memory instrumentation
  * pyg-lib / num_workers / persistent_workers / epoch checkpointing
    (LI-Large-scale concerns, irrelevant at HI-Small size)

Output CSVs (gargaml_tree.py's naming, with a "_graphsage" suffix):
  * results/<dataset>_AUC_PR_graphsage_undirected_graphsage_combined.csv (+ the
    other LEGACY_METRICS, + their _std_combined.csv companions)
  * results/<dataset>_imbalance_undirected_graphsage_combined.csv
  * results/<dataset>_undirected_graphsage_metrics.csv -- the tidy frame
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd
import torch

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.methods.graphsage import build_graph_data, get_device, score_all, train_fold
from src.utils.evaluation import (
    evaluate_scores,
    metric_records,
    nan_metrics,
    write_metrics,
)

MODEL_KEY = "graphsage_u"  # see src/utils/naming.py
FEATURES = "topology"       # the spec's config A ("topology-only, strict parity")

# Small on purpose -- extend the same way gargaml_tree.py's CUT_OFFS/TARGET_COLUMNS
# are, once a run at this scale is confirmed to behave sensibly.
CUT_OFFS = [0.1]
TARGET_COLUMNS = ['Is Laundering']
EPOCHS = 10

# No separate N_FOLDS constant here on purpose: the fold count and fold values
# actually trained on are read straight from results/<dataset>_folds.csv (whatever
# gargaml_tree.py's own N_FOLDS was set to when it wrote that file). A separate
# constant here could silently drift out of sync with the file -- e.g. testing with
# a 2-fold partition but forgetting to also flip a local N_FOLDS=2, which would
# quietly loop the wrong fold indices instead of failing loudly. To test with fewer
# folds, regenerate the folds file with gargaml_tree.py's N_FOLDS=2 (see CLAUDE.md
# "How to run things") -- this script adapts automatically.


def load_labels(dataset):
    """Per-account label table, aligned by account (same functions gargaml_tree.py uses)."""
    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans="data/"+dataset+"_Trans.csv",
        path_patterns="data/"+dataset+"_Patterns.txt",
    )
    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended, pattern_columns)
    return laundering_combined


def run_target(dataset, data, node_order, laundering_combined, folds_df, cutoff, target,
               device):
    context = dict(
        dataset=dataset, direction="undirected", features=FEATURES,
        cutoff=cutoff, target=target, seed=1997,
    )

    fold_lookup = folds_df[(folds_df["cutoff"] == cutoff) & (folds_df["target"] == target)]
    if fold_lookup.empty:
        # Task 7 already skips (cutoff, target) pairs with too few positives for
        # N_FOLDS-fold CV; a missing pair here just means task 7 skipped it too.
        reason = "no fold partition for this (cutoff, target) -- see results/"+dataset+"_folds.csv"
        print("    "+reason)
        return metric_records(nan_metrics(), status="skipped: "+reason, model=MODEL_KEY,
                               fold=np.nan, **context)

    fold_by_account = fold_lookup.set_index("account")["fold"]

    y_series = (laundering_combined[target] > cutoff).astype(int).reindex(node_order)
    if y_series.isna().any():
        # Every graph node comes from the same transaction CSV summarise_ML_labels
        # groups over, so this should not happen -- surfaced loudly rather than
        # silently coercing to a label.
        raise ValueError(f"{y_series.isna().sum()} graph nodes have no label for "
                          f"target={target!r}; account universes have diverged.")
    y = y_series.values.astype(np.float32)

    fold_series = fold_by_account.reindex(node_order)
    if fold_series.isna().any():
        raise ValueError(f"{fold_series.isna().sum()} graph nodes are missing from "
                          f"the folds file for cutoff={cutoff}, target={target!r}.")
    fold_of_node = fold_series.values.astype(int)

    folds = sorted(fold_lookup["fold"].unique())
    print(f"    {len(folds)}-fold partition found in the folds file: {folds}")

    records = []
    oof_scores = np.full(len(node_order), np.nan)

    for fold in folds:
        test_mask = torch.tensor(fold_of_node == fold)
        train_mask = ~test_mask

        n_test = int(test_mask.sum())
        n_pos = int(y[test_mask.numpy()].sum())
        fold_context = dict(context, fold=fold)

        print(f"  fold {fold}: {int(train_mask.sum())} train / {n_test} test")
        model = train_fold(data, train_mask, y, device, epochs=EPOCHS)
        scores = score_all(model, data, device)
        oof_scores[test_mask.numpy()] = scores[test_mask.numpy()]

        y_test = y[test_mask.numpy()]
        y_score = scores[test_mask.numpy()]
        metrics = evaluate_scores(y_test, y_score, y_pred=(y_score > 0.5).astype(int))
        records += metric_records(metrics, status="ok", n_test=n_test, n_pos=n_pos,
                                   model=MODEL_KEY, **fold_context)

    # Pooled out-of-fold pass (fold=-1): the alert-queue headline numbers, ranked
    # over every account -- same convention as gargaml_tree.py's CV branch.
    pooled_context = dict(context, fold=-1)
    if np.isnan(oof_scores).any():
        metrics, status = nan_metrics(), "skipped: incomplete out-of-fold coverage"
    else:
        metrics = evaluate_scores(y, oof_scores, y_pred=(oof_scores > 0.5).astype(int))
        status = "ok"
    records += metric_records(metrics, status=status, n_test=len(y), n_pos=int(y.sum()),
                               model=MODEL_KEY, **pooled_context)

    return records


def main():
    dataset = "HI-Small"
    device = get_device()
    print(f"device: {device}")

    print("Building graph (cached after first run)...")
    data, node_order = build_graph_data(dataset)
    print(f"  {len(node_order)} nodes, {data.num_edges} directed entries, "
          f"{data.num_node_features} features")

    laundering_combined = load_labels(dataset)

    folds_path = f"results/{dataset}_folds.csv"
    if not os.path.exists(folds_path):
        raise FileNotFoundError(
            f"{folds_path} not found -- run scripts/gargaml_tree.py with N_FOLDS >= 2 "
            "first (task 7); GraphSAGE reads that partition rather than re-deriving it."
        )
    folds_df = pd.read_csv(folds_path)

    records = []
    for cutoff in CUT_OFFS:
        for target in TARGET_COLUMNS:
            print(f"\n=== {dataset} (undirected), GraphSAGE: cutoff={cutoff} target={target} ===")
            records += run_target(dataset, data, node_order, laundering_combined,
                                   folds_df, cutoff, target, device)

    write_metrics(records, dataset, "undirected", suffix="_graphsage", write_std=True)


if __name__ == "__main__":
    main()
