"""
GraphSAGE baseline (task 1) -- the GNN comparison R2 asked for.

A 2-layer GraphSAGE has the same receptive field as GARG-AML's second-order
neighbourhood: same information, one model interpretable and O(|V|), the
other needing a GPU and neighbour sampling. Everything here exists to make
that comparison fair rather than rhetorical.

  * **Both feature configurations**, reported side by side: ``topology``
    (strict parity with GARG-AML's inputs) and ``attributes`` (deliberately
    generous -- amounts, counts, currencies, banks, timing). Neither ever
    sees a GARG-AML score, block density or block size.
  * **The same folds as the tree models**, read from task 7's persisted
    ``results/<dataset>_folds.csv`` rather than re-derived, so the two
    models are paired fold by fold instead of compared across independently
    drawn splits.
  * **Early stopping on validation AUC-PR**, on a stratified 10% slice
    carved out of the *training* fold. The test fold is never touched.
  * **Timing and peak memory recorded separately** for preprocessing, fit
    and inference. This is the scalability argument, and a single wall-clock
    number would not support it: GARG-AML's fit cost is zero, which is only
    visible if fit is reported apart from preprocessing.

Run budget (task 1's own arithmetic: ~90 runs + ~30)
----------------------------------------------------
HI-Small  3 cut-offs x 3 targets x 5 folds x 2 configs = 90 fits.
LI-Large  3 cut-offs x 1 target  x 5 folds x 2 configs = 30 fits.

The cut-offs are the paper's headline 0.1 / 0.5 / 0.9 and the HI-Small
targets are the pooled label plus the two patterns GARG-AML actually targets
(GATHER-SCATTER, SCATTER-GATHER). This is a **deliberate, logged** reduction
of gargaml_tree.py's 5 x 9 grid, not a silent truncation: a GNN fit is
minutes where a decision tree is seconds, and the full grid would be ~450
fits. Widen ``DATASETS`` below if the compute budget allows.

Expect cut-off 0.9 to fail on some targets for too few positives, exactly as
it does for the tree models. Those cells are reported as NaN with a reason,
never dropped.

Outputs (per feature config)
----------------------------
  * ``results/<dataset>_undirected<suffix>_metrics.csv`` -- the tidy frame,
    one row per (config, cut-off, target, fold, metric), including the
    timing and memory rows
  * ``results/<dataset>_<metric>_graphsage_undirected<suffix>_combined.csv``
    and their ``_std_`` companions -- the historical matrix format
  * ``results/<dataset>_undirected<suffix>_feature_schema.csv`` -- task 10
  * ``results/<dataset>_graphsage_tidy.csv`` -- both configs in one tidy
    frame, including the per-config preprocessing timings
  * ``results/<dataset>_graphsage_runs.csv`` -- the spec's deliverable: one
    row per (dataset, config, cut-off, target, fold) with every metric,
    the timings, peak memory, epochs to early stop and status
  * ``results/<dataset>_graphsage_summary.csv`` -- mean/std across folds via
    task 7's shared ``aggregate_folds``
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
from src.data.bank_views import (bank_clients, parse_view, patterns_path,
                                resolve_banks, trans_path)
from src.methods.graphsage import (
    FEATURE_CONFIGS,
    build_graph_data,
    feature_schema,
    get_device,
    peak_gpu_memory_mb,
    peak_host_memory_mb,
    reset_peak_gpu_memory,
    timed_score_all,
    train_fold,
)
from src.utils.evaluation import (
    SEED,
    aggregate_folds,
    evaluate_scores,
    metric_records,
    metrics_frame,
    nan_metrics,
    write_metrics,
)
from src.utils.naming import pretty_config
from src.utils.runtime import env_override, select_datasets, echo_config, as_list

MODEL_KEY = "graphsage_u"  # see src/utils/naming.py

# Result-file suffix per feature config. Config A keeps the plain
# "_graphsage" suffix the first pass wrote, so files already on disk and any
# notebook pointing at them stay valid.
CONFIG_SUFFIXES = {"topology": "_graphsage", "attributes": "_graphsage_attr"}

DATASETS = {
    "HI-Small": dict(
        cut_offs=[0.1, 0.5, 0.9],
        targets=["Is Laundering", "GATHER-SCATTER", "SCATTER-GATHER"],
        infer_batch_size=None,  # the full graph fits; full-graph inference
        num_workers=0,
    ),
    "LI-Large": dict(
        cut_offs=[0.1, 0.5, 0.9],
        targets=["Is Laundering"],
        infer_batch_size=4096,  # 2M nodes / 352M edges will not fit on a GPU
        num_workers=4,          # with pyg-lib installed; see task 1's notes
    ),
}

# May be a plain dataset or a task-5 single-bank view, e.g. "HI-Small_bank012".
# A view reads data/HI-Small_Trans.csv filtered to that bank, evaluates the
# bank's own clients, and needs its own results/<view>_folds.csv -- so run
# gargaml_tree.py on the same view name first.
#
# One dataset per invocation, deliberately: a GraphSAGE sweep is a job of its
# own on VSC, so the views are run by setting this and resubmitting rather
# than by looping here. The partial-observability views (task 5) are
# "HI-Small_bank012" and "HI-Small_banktop50"; see
# notebooks/BankObservability.ipynb for why those two.
DATASET = "HI-Small"
CONFIGS = list(FEATURE_CONFIGS)  # topology (A), attributes (B)

EPOCHS = 50       # an upper bound; early stopping decides the real number
PATIENCE = 5      # epochs without a validation AUC-PR improvement
CHECKPOINT_DIR = "results/checkpoints"

# Slurm overrides; the constants above remain the documented defaults.
# CHECKPOINT_DIR defaults to $VSC_SCRATCH on the cluster: checkpoints are
# transient per-epoch state, and $VSC_DATA is the quota'd volume. Nothing
# downstream reads them -- only train_fold's own resume does.
DATASET = env_override("dataset", DATASET)
CONFIGS = env_override("configs", CONFIGS, as_list)
CHECKPOINT_DIR = env_override(
    "checkpoint_dir",
    os.path.join(os.environ["VSC_SCRATCH"], "gargaml", "checkpoints")
    if os.environ.get("VSC_SCRATCH") else CHECKPOINT_DIR)

# No N_FOLDS constant on purpose: the fold count and the fold values actually
# trained on are read from results/<dataset>_folds.csv, whatever
# gargaml_tree.py's N_FOLDS was when it wrote that file. A local constant
# could drift out of sync with the file -- testing against a 2-fold partition
# while looping five fold indices would fail quietly rather than loudly. To
# test with fewer folds, regenerate the file with N_FOLDS = 2 (CLAUDE.md,
# "How to run things"); this script adapts on its own.


def load_labels(dataset):
    """Per-account label table -- the same functions gargaml_tree.py uses.

    For a task-5 bank view this is restricted to the bank's own clients, so
    the evaluated population matches gargaml_tree.py's exactly and the two
    models stay comparable fold for fold.
    """
    _, banks = parse_view(dataset)
    # Expand a group spec ("top50") before bank_clients filters on it; see
    # src/data/bank_views.py. src.methods.graphsage resolves the same way,
    # so the graph and the evaluated population stay in agreement.
    banks = resolve_banks(banks, trans_path(dataset))
    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans=trans_path(dataset),
        path_patterns=patterns_path(dataset),
        banks=banks,
    )
    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended, pattern_columns)

    if banks is not None:
        clients = bank_clients(transactions_df_extended, banks)
        laundering_combined = laundering_combined[laundering_combined.index.isin(clients)]

    return laundering_combined


def evaluable_mask(laundering_combined, node_order):
    """Which graph nodes are part of the evaluated population.

    All of them on the full graph. In a bank view the graph also holds the
    external counterparties that remain of each client's neighbourhood: they
    are needed for message passing -- dropping them would defeat the point of
    the experiment -- but the bank has no labels for them and they are not
    scored. Everything below masks with this rather than with ``~test_mask``.
    """
    return pd.Index(node_order).isin(laundering_combined.index).astype(bool)


def align_to_graph(series, node_order, what, eval_mask=None):
    """Reindex an account-keyed series onto the graph's node order.

    Raises rather than filling: every graph node comes from the same
    transaction file the labels are grouped over, so a gap means the two
    account universes have diverged, and quietly defaulting it would put a
    fabricated label or fold into a published number.

    ``eval_mask`` narrows that check to the evaluated population, which is
    what a task-5 bank view needs: its external counterparties are unlabelled
    by construction, not by divergence. Their entries come back NaN for the
    caller to fill with an inert value; they are masked out of every fit and
    every metric, so the fill is never read.
    """
    aligned = series.reindex(node_order)
    missing = aligned.isna()
    if eval_mask is not None:
        missing = missing & eval_mask
    if missing.any():
        raise ValueError(str(int(missing.sum()))+" evaluated nodes have no "+what+
                         "; the account universes have diverged.")
    return aligned


def checkpoint_path(dataset, config, cutoff, target, fold):
    """One checkpoint per fold, so a wall-clock kill costs one epoch."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    safe_target = target.replace(" ", "-")
    return (CHECKPOINT_DIR+"/"+dataset+"_"+config+"_"+str(cutoff)+"_"+safe_target+
            "_fold"+str(fold)+".pt")


def run_target(dataset, config, data, node_order, laundering_combined, folds_df,
               cutoff, target, device, settings, eval_mask=None):
    """Train and evaluate every fold for one (cut-off, target) cell."""
    context = dict(dataset=dataset, direction="undirected", features=config,
                   cutoff=cutoff, target=target, seed=SEED)

    fold_lookup = folds_df[(folds_df["cutoff"] == cutoff) & (folds_df["target"] == target)]
    if fold_lookup.empty:
        # Task 7 skips (cut-off, target) pairs with too few positives to fold;
        # a missing pair here means it skipped that pair too.
        reason = "no fold partition for this (cutoff, target) -- see results/"+dataset+"_folds.csv"
        print("    "+reason)
        return metric_records(nan_metrics(), status="skipped: "+reason, model=MODEL_KEY,
                              fold=np.nan, **context)

    if eval_mask is None:
        eval_mask = np.ones(len(node_order), dtype=bool)

    # 0 and -1 are inert fills for the nodes outside the evaluated population
    # (a bank view's external counterparties): eval_mask removes them from
    # every fit, every metric and the pooled pass below, so neither value is
    # ever read. They exist only so these stay plain numeric arrays.
    y = align_to_graph((laundering_combined[target] > cutoff).astype(int),
                       node_order, "label for target "+repr(target), eval_mask=eval_mask)
    y = y.fillna(0).values.astype(np.float32)

    fold_of_node = align_to_graph(fold_lookup.set_index("account")["fold"],
                                  node_order, "fold assignment",
                                  eval_mask=eval_mask).fillna(-1).values.astype(int)

    folds = sorted(fold_lookup["fold"].unique())
    print("    "+str(len(folds))+"-fold partition from the folds file: "+str(folds))

    records = []
    oof_scores = np.full(len(node_order), np.nan)

    for fold in folds:
        test_mask = (fold_of_node == fold) & eval_mask
        train_mask = (fold_of_node != fold) & eval_mask
        n_test, n_pos = int(test_mask.sum()), int(y[test_mask].sum())
        fold_context = dict(context, fold=fold)

        print("  fold "+str(fold)+": "+str(int(train_mask.sum()))+" train / "+str(n_test)
              +" test ("+str(n_pos)+" positive)")

        if n_pos == 0 or int(y[train_mask].sum()) == 0:
            # No positives to learn from or to score against: AUC-PR is
            # undefined. Same failure mode the tree models hit at cut-off 0.9.
            reason = "no positives in the train or test fold"
            print("    skipped: "+reason)
            records += metric_records(nan_metrics(), status="skipped: "+reason,
                                      n_test=n_test, n_pos=n_pos, model=MODEL_KEY,
                                      **fold_context)
            continue

        reset_peak_gpu_memory(device)
        model, info = train_fold(
            data, train_mask, y, device, epochs=EPOCHS, patience=PATIENCE,
            num_workers=settings["num_workers"], seed=SEED,
            checkpoint_path=checkpoint_path(dataset, config, cutoff, target, fold),
        )
        scores, infer_seconds = timed_score_all(
            model, data, device, batch_size=settings["infer_batch_size"],
            num_workers=settings["num_workers"])

        oof_scores[test_mask] = scores[test_mask]

        metrics = evaluate_scores(y[test_mask], scores[test_mask],
                                  y_pred=(scores[test_mask] > 0.5).astype(int))
        # Status stays exactly "ok": aggregate_folds counts n_folds_ok by
        # equality against that string, so a more descriptive status here
        # would drop a perfectly good fold out of the count. Whether the
        # fold had a validation slice is recorded as data, below.
        records += metric_records(metrics, status="ok", n_test=n_test, n_pos=n_pos,
                                  model=MODEL_KEY, **fold_context)

        # Timing and memory travel in the same tidy frame as the metrics, so
        # aggregate_folds reduces them across folds for free and the
        # scalability table is a filter rather than a second pipeline.
        records += metric_records({
            "fit_seconds": info["fit_seconds"],
            "infer_seconds": infer_seconds,
            "epochs_run": info["epochs_run"],
            "epochs_to_best": info["epochs_to_best"],
            "best_val_AP": info["best_val_AP"],
            # Process high-water mark, not this fold in isolation: RSS only
            # ever rises, so read it as "peak so far" rather than per-fold.
            "peak_host_mb": peak_host_memory_mb(),
            "peak_gpu_mb": peak_gpu_memory_mb(device),
            # 0 means the fold had too few positives to carve a stratified
            # validation slice, so it trained for a fixed EPOCHS with no
            # early stopping. Reported rather than hidden in a status string.
            "validated": float(info["validated"]),
        }, status="ok", n_test=n_test, n_pos=n_pos, model=MODEL_KEY, **fold_context)

    # Pooled out-of-fold pass (fold=-1): every account scored by a model that
    # never trained on it, ranked over the whole population. These are the
    # alert-queue headline numbers -- same convention as gargaml_tree.py.
    pooled_context = dict(context, fold=-1)
    # Only the evaluated population is ranked: in a bank view the external
    # counterparties were never scored, and leaving their NaNs in would read
    # as incomplete coverage and skip the cell.
    pooled_scores, pooled_y = oof_scores[eval_mask], y[eval_mask]
    if np.isnan(pooled_scores).any():
        metrics, status = nan_metrics(), "skipped: incomplete out-of-fold coverage"
    else:
        metrics = evaluate_scores(pooled_y, pooled_scores,
                                  y_pred=(pooled_scores > 0.5).astype(int))
        status = "ok"
    records += metric_records(metrics, status=status, n_test=len(pooled_y),
                              n_pos=int(pooled_y.sum()), model=MODEL_KEY, **pooled_context)

    return records


def wide_runs_frame(long_df):
    """The spec's deliverable: one row per run, one column per metric.

    The tidy frame is the source of truth; this is a view of it. ``P@100``
    and friends are folded back into single column names so the table reads
    like the results tables in the manuscript.
    """
    df = long_df.copy()
    has_k = df["K"].notna()
    df["column"] = np.where(
        has_k,
        df["metric"].str.replace("@K", "", regex=False)+"@"+df["K"].fillna(0).astype(int).astype(str),
        df["metric"],
    )
    index = ["dataset", "direction", "model", "features", "cutoff", "target", "seed", "fold"]
    wide = df.pivot_table(index=index, columns="column", values="value", aggfunc="first")

    # status/n_test/n_pos are per run, not per metric -- carry one copy.
    meta = df.groupby(index)[["n_test", "n_pos"]].first()
    status = df.groupby(index)["status"].agg(lambda s: sorted(set(s))[0])
    return wide.join(meta).join(status).reset_index()


def main():
    dataset = DATASET
    # DATASETS is keyed by the underlying dataset, so a task-5 view
    # ("HI-Small_bank012") inherits its base dataset's sweep and batching
    # settings instead of needing a duplicated entry.
    settings = DATASETS[parse_view(dataset)[0]]
    device = get_device()

    print("device: "+str(device))
    print("Transductive evaluation on task 7's persisted folds; neighbour "
          "features are built on the full graph before folding (R2-M6).")

    folds_path = "results/"+dataset+"_folds.csv"
    if not os.path.exists(folds_path):
        raise FileNotFoundError(
            folds_path+" not found -- run scripts/gargaml_tree.py with N_FOLDS >= 2 "
            "first (task 7); GraphSAGE reads that partition rather than re-deriving it.")
    folds_df = pd.read_csv(folds_path)

    laundering_combined = load_labels(dataset)

    all_records = []
    eval_mask = None  # built once the node order is known, below
    for config in CONFIGS:
        suffix = CONFIG_SUFFIXES[config]
        print("\n=== "+dataset+", "+pretty_config(MODEL_KEY, config)+" ===")

        data, node_order, preprocess_seconds = build_graph_data(dataset, config=config)
        print("  "+str(len(node_order))+" nodes, "+str(data.num_edges)+" directed entries, "
              +str(data.num_node_features)+" features, built in "
              +format(preprocess_seconds, ".1f")+" s")

        # Both configs share one node order (same structure, different
        # features), so this is computed once and reused.
        if eval_mask is None:
            eval_mask = evaluable_mask(laundering_combined, node_order)
            if not eval_mask.all():
                print("  bank view: "+str(int(eval_mask.sum()))+" of "+str(len(node_order))
                      +" nodes are clients and scored; the rest carry messages only")

        schema_path = "results/"+dataset+"_undirected"+suffix+"_feature_schema.csv"
        feature_schema(config).to_csv(schema_path, index=False)
        print("  feature schema -> "+schema_path)

        # Preprocessing is per (dataset, config), not per fold, so it is one
        # row rather than a value repeated across every fold as if it had been
        # paid each time. Kept out of the records that reach write_metrics:
        # that function takes the row and column order of the legacy matrices
        # from every row it is given, so a cutoff=NaN / target="(all)" row
        # would add a spurious row and column to each published matrix file.
        preprocess_records = metric_records(
            {"preprocess_seconds": preprocess_seconds,
             "peak_host_mb_after_preprocess": peak_host_memory_mb()},
            status="ok", model=MODEL_KEY, dataset=dataset, direction="undirected",
            features=config, cutoff=np.nan, target="(all)", seed=SEED, fold=np.nan)

        records = []

        for cutoff in settings["cut_offs"]:
            for target in settings["targets"]:
                print("\n--- cutoff="+str(cutoff)+"  target="+target+" ---")
                records += run_target(dataset, config, data, node_order,
                                      laundering_combined, folds_df, cutoff, target,
                                      device, settings, eval_mask=eval_mask)

        write_metrics(records, dataset, "undirected", suffix=suffix, write_std=True)
        all_records += preprocess_records + records

    long_df = metrics_frame(all_records)

    # The complete record, both configs together, including the per-config
    # preprocessing rows. Those carry cutoff=NaN and fold=NaN, which
    # pivot_table drops as index keys and aggregate_folds filters out, so
    # without this file the preprocessing timings would exist in no output
    # at all.
    tidy_path = "results/"+dataset+"_graphsage_tidy.csv"
    long_df.to_csv(tidy_path, index=False)
    print("\ntidy metrics  -> "+tidy_path)

    runs_path = "results/"+dataset+"_graphsage_runs.csv"
    wide_runs_frame(long_df).to_csv(runs_path, index=False)
    print("per-run table -> "+runs_path)

    summary_path = "results/"+dataset+"_graphsage_summary.csv"
    aggregate_folds(long_df).to_csv(summary_path, index=False)
    print("fold summary  -> "+summary_path)


if __name__ == "__main__":
    echo_config(__file__, dataset=DATASET, configs=CONFIGS,
                checkpoint_dir=CHECKPOINT_DIR, epochs=EPOCHS, patience=PATIENCE)
    main()
