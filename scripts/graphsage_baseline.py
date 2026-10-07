"""
GraphSAGE baseline for comparison with GARG-AML.

A 2-layer GraphSAGE has the same receptive field as GARG-AML's second-order
neighbourhood. Every feature is built from the raw transaction file, so no
GARG-AML score, block density or block size enters the model.

  * Two feature configurations, reported side by side: ``topology`` (degree
    and log-degree, matching GARG-AML's inputs) and ``attributes``
    (amounts, counts, currencies, banks, timing).
  * The folds are read from ``results/<dataset>_folds.csv``, the same
    partition the tree models use, so the two are paired fold by fold.
  * Early stopping on validation AUC-PR, measured on a stratified 10% slice
    carved out of the training fold. The test fold is never touched.
  * Timing and peak memory recorded separately for preprocessing, fit and
    inference, so that fit cost is visible apart from preprocessing.

Sweep
-----
The tree models' grid (gargaml_tree.py), cell for cell:

HI-Small  6 cut-offs x 9 targets x 5 folds x 2 configs = 540 fits.
LI-Large  4 cut-offs x 9 targets x 5 folds x 2 configs = 360 fits.

LI-Large runs the headline slice ``HEADLINE_CUTOFFS`` only, as the tree grid
does. The first runs were a logged reduction of this (headline cut-offs, three
HI-Small targets, "Is Laundering" alone on LI-Large); ``GARGAML_CUTOFFS`` and
``GARGAML_TARGETS`` narrow a run to any subset.

A cell with too few positives to fit -- cut-off 0.9 on some targets -- is
reported as NaN with a reason, never dropped.

Running in parallel
-------------------
A full sweep is far too long for one GPU job, and its targets are independent,
so it runs as one job per target (``GARGAML_TARGETS``), each writing to its own
``GARGAML_OUTPUT_DIR`` under a common parent. ``GARGAML_MERGE_FROM=<parent>``
then combines them into the files one full sweep would have written, in its
cut-off x target order (slurm/README.md has the commands). Run the prep job
first: without the caches, every job builds and writes the same cache files
at once.

Preparing on a CPU node
-----------------------
``GARGAML_PREPARE_ONLY=1`` builds and caches everything the fits read -- the
per-account labels, the graph structure and both feature matrices -- and
exits without fitting. It needs no GPU and no folds file. The label build is
the baseline's host-memory peak (166.6 GiB MaxRSS on LI-Large, measured on
label_distribution.py's identical call, job 62206918), while wICE's gpu_a100
partition allows at most 126,000 MiB per GPU. So LI-Large is prepared in a
CPU job (slurm/graphsage_prep.slurm) and the GPU job only loads the caches.
A preprocessing time measured in the GPU job is then a cache load: the cold
build times are in ``<dataset>_graphsage_prep.csv``, and every preprocessing
row carries ``preprocess_from_cache``.

Outputs (per feature config)
----------------------------
  * ``results/<dataset>_undirected<suffix>_metrics.csv`` -- the tidy frame,
    one row per (config, cut-off, target, fold, metric), including the
    timing and memory rows
  * ``results/<dataset>_<metric>_graphsage_undirected<suffix>_combined.csv``
    and their ``_std_`` companions -- the matrix format
  * ``results/<dataset>_undirected<suffix>_feature_schema.csv`` -- the
    feature schema of the configuration
  * ``results/<dataset>_graphsage_tidy.csv`` -- both configs in one tidy
    frame, including the per-config preprocessing timings
  * ``results/<dataset>_graphsage_runs.csv`` -- one row per (dataset,
    config, cut-off, target, fold) with every metric, the timings, peak
    memory, epochs to early stop and status
  * ``results/<dataset>_graphsage_summary.csv`` -- mean/std across folds

Caches (per dataset, shared by both configs where they can be)
-------------------------------------------------------------
  * ``results/<dataset>_graphsage_labels.pkl`` -- the per-account label table
  * ``results/<dataset>_graphsage_structure.pt`` and
    ``results/<dataset>_graphsage_x_<config>.pt`` -- structure and features
  * ``results/<dataset>_graphsage_prep.csv`` -- prepare-only runs: seconds and
    peak host memory per step, and whether the step was a cache load

None of the caches is ever invalidated; delete one to rebuild it.
"""

import glob
import os
import sys
import time

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
    graph_cache_paths,
    peak_gpu_memory_mb,
    peak_host_memory_mb,
    reset_peak_gpu_memory,
    timed_score_all,
    train_fold,
)
from src.utils.evaluation import (
    CUT_OFFS,
    HEADLINE_CUTOFFS,
    SEED,
    aggregate_folds,
    evaluate_scores,
    metric_records,
    metrics_frame,
    nan_metrics,
    write_metrics,
)
from src.utils.naming import pretty_config
from src.utils.runtime import (env_override, select_datasets, echo_config, as_list,
                              as_bool, resolve_results_dir, write_csv)

MODEL_KEY = "graphsage_u"  # see src/utils/naming.py

# Result-file suffix per feature config.
CONFIG_SUFFIXES = {"topology": "_graphsage", "attributes": "_graphsage_attr"}

# The same nine targets as gargaml_tree.py's TARGET_COLUMNS, in its order.
TARGET_COLUMNS = ["Is Laundering", "FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER",
                  "CYCLE", "RANDOM", "BIPARTITE", "STACK"]

DATASETS = {
    "HI-Small": dict(
        cut_offs=CUT_OFFS,
        targets=TARGET_COLUMNS,
        infer_batch_size=None,  # the full graph fits; full-graph inference
        num_workers=0,
    ),
    "LI-Large": dict(
        cut_offs=HEADLINE_CUTOFFS,
        targets=TARGET_COLUMNS,
        infer_batch_size=4096,  # the full graph will not fit on a GPU
        num_workers=4,          # with pyg-lib installed
    ),
}

# May be a plain dataset or a single-bank view, e.g. "HI-Small_bank012" -- a
# view evaluates that bank's own clients and needs its own
# results/<view>_folds.csv, so run gargaml_tree.py on the same view name
# first. One dataset per invocation: set this and resubmit for a view.
DATASET = "HI-Small"
CONFIGS = list(FEATURE_CONFIGS)  # topology, attributes

EPOCHS = 50       # an upper bound; early stopping decides the real number
PATIENCE = 5      # epochs without a validation AUC-PR improvement
CHECKPOINT_DIR = "results/checkpoints"

# Slurm overrides. CHECKPOINT_DIR defaults to $VSC_SCRATCH on the cluster --
# checkpoints are transient per-epoch state, unlike the quota'd $VSC_DATA --
# and nothing downstream reads them except train_fold's own resume.
DATASET = env_override("dataset", DATASET)
# GARGAML_EPOCHS raises the cap for a rerun of cells that hit it (the attributes
# config at cut-off 0.0 ran all 50 epochs with its best validation AUC-PR at
# the last one). Checkpoints of a non-default cap get their own file names
# (checkpoint_path), so a rerun trains from scratch and its fit_seconds is the
# whole fit, not the continuation of a 50-epoch checkpoint.
EPOCHS = env_override("epochs", EPOCHS, int)
# GARGAML_CUTOFFS narrows the sweep to some of the dataset's cut-offs.
CUTOFFS_OVERRIDE = env_override("cutoffs", None, lambda raw: [float(c) for c in as_list(raw)])
# GARGAML_TARGETS replaces the dataset's target list -- one target per job is
# how a sweep runs in parallel. It is checked against the targets the folds
# file was split on rather than against DATASETS, so it may also widen it.
TARGETS_OVERRIDE = env_override("targets", None, as_list)
# GARGAML_MERGE_FROM=<parent>: combine the per-target runs under <parent> and
# exit without fitting -- see "Running in parallel" above.
MERGE_FROM = env_override("merge_from", None)
CONFIGS = env_override("configs", CONFIGS, as_list)
RESULTS_DIR = resolve_results_dir()
# Everything this script *reads* (folds, label / graph / feature caches) stays in
# RESULTS_DIR. GARGAML_OUTPUT_DIR redirects only what it *writes* -- the tidy
# metrics, matrices, schema and the runs / summary files -- so a partial rerun
# cannot overwrite the full sweep, and reporting.load_metrics (which globs
# RESULTS_DIR/*_metrics.csv) does not count the rerun's rows twice.
OUTPUT_DIR = env_override("output_dir", RESULTS_DIR)
CHECKPOINT_DIR = env_override(
    "checkpoint_dir",
    os.path.join(os.environ["VSC_SCRATCH"], "gargaml", "checkpoints")
    if os.environ.get("VSC_SCRATCH") else RESULTS_DIR+"/checkpoints")

# GARGAML_PREPARE_ONLY=1: build the label, structure and feature caches and
# exit without fitting -- see "Preparing on a CPU node" above.
PREPARE_ONLY = env_override("prepare_only", False, as_bool)

# No N_FOLDS constant: the fold count and the fold values trained on are read
# from results/<dataset>_folds.csv, so a local constant cannot drift out of
# sync with the partition on disk.

# That partition only ever exists for the IBM data: cross-validation is scoped
# there and the synthetic grid keeps its single holdout split, so a synthetic
# name arriving via GARGAML_DATASET is a mistake rather than a run to attempt.
if DATASET.startswith("synthetic"):
    raise ValueError(
        "this baseline is for the IBM data only; " + DATASET + " is synthetic. "
        "The synthetic grid is not cross-validated and has no folds file."
    )


def labels_cache_path(dataset):
    """Where ``load_labels`` caches the per-account label table."""
    return RESULTS_DIR+"/"+dataset+"_graphsage_labels.pkl"


def load_labels(dataset):
    """Per-account label table -- the same functions gargaml_tree.py uses.

    For a bank view this is restricted to the bank's own clients, so the
    evaluated population matches gargaml_tree.py's and the two models stay
    comparable fold for fold.

    Cached at ``labels_cache_path``: the table is identical for every config,
    cut-off and fold, and building it is the baseline's host-memory peak.
    Pickled rather than CSV so the account index round-trips exactly.
    """
    cache_path = labels_cache_path(dataset)
    if os.path.exists(cache_path):
        print("labels <- "+cache_path)
        return pd.read_pickle(cache_path)

    _, banks = parse_view(dataset)
    # Expand a group spec ("top50") before bank_clients filters on it;
    # src.methods.graphsage resolves the same way, so the graph and the
    # evaluated population stay in agreement.
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

    # Temp file + os.replace, so a job killed mid-write cannot leave a
    # truncated pickle for the next run to load.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = cache_path+".tmp"+str(os.getpid())
    laundering_combined.to_pickle(tmp)
    os.replace(tmp, cache_path)
    print("labels -> "+cache_path)

    return laundering_combined


def evaluable_mask(laundering_combined, node_order):
    """Which graph nodes are part of the evaluated population.

    All of them on the full graph. In a bank view the graph also holds the
    external counterparties that remain of each client's neighbourhood: they
    carry messages, but the bank has no labels for them and they are not
    scored. Everything below masks with this rather than with ``~test_mask``.
    """
    return pd.Index(node_order).isin(laundering_combined.index).astype(bool)


def align_to_graph(series, node_order, what, eval_mask=None):
    """Reindex an account-keyed series onto the graph's node order.

    Raises rather than filling: every graph node comes from the same
    transaction file the labels are grouped over, so a gap means the two
    account universes have diverged, and quietly defaulting it would put a
    fabricated label or fold into a reported number.

    ``eval_mask`` narrows that check to the evaluated population, which is
    what a bank view needs: its external counterparties are unlabelled by
    construction, not by divergence. Their entries come back NaN for the
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
    cap = "" if EPOCHS == 50 else "_ep"+str(EPOCHS)
    return (CHECKPOINT_DIR+"/"+dataset+"_"+config+"_"+str(cutoff)+"_"+safe_target+
            "_fold"+str(fold)+cap+".pt")


def run_target(dataset, config, data, node_order, laundering_combined, folds_df,
               cutoff, target, device, settings, eval_mask=None):
    """Train and evaluate every fold for one (cut-off, target) cell."""
    context = dict(dataset=dataset, direction="undirected", features=config,
                   cutoff=cutoff, target=target, seed=SEED)

    fold_lookup = folds_df[(folds_df["cutoff"] == cutoff) & (folds_df["target"] == target)]
    if fold_lookup.empty:
        # A (cut-off, target) pair with too few positives to fold has no
        # partition in the folds file.
        reason = "no fold partition for this (cutoff, target) -- see results/"+dataset+"_folds.csv"
        print("    "+reason)
        return metric_records(nan_metrics(), status="skipped: "+reason, model=MODEL_KEY,
                              fold=np.nan, **context)

    if eval_mask is None:
        eval_mask = np.ones(len(node_order), dtype=bool)

    # 0 and -1 are inert fills for nodes outside the evaluated population (a
    # bank view's external counterparties): eval_mask removes them from every
    # fit, every metric and the pooled pass below, so neither value is ever
    # read. They exist only so these stay plain numeric arrays.
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
            # undefined.
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
        # equality against that string. Whether the fold had a validation
        # slice is recorded as data, below.
        records += metric_records(metrics, status="ok", n_test=n_test, n_pos=n_pos,
                                  model=MODEL_KEY, **fold_context)

        # Timing and memory travel in the same tidy frame as the metrics, so
        # aggregate_folds reduces them across folds and the scalability table
        # is a filter rather than a second pipeline.
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
    """One row per run, one column per metric.

    The tidy frame is the source of truth; this is a view of it. ``P@100``
    and friends are folded back into single column names.
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


def sweep_settings(dataset):
    """``DATASETS``' entry for ``dataset``, with the cut-off and target overrides."""
    # DATASETS is keyed by the underlying dataset, so a view
    # ("HI-Small_bank012") inherits its base dataset's sweep and batching
    # settings instead of needing a duplicated entry.
    settings = dict(DATASETS[parse_view(dataset)[0]])
    if CUTOFFS_OVERRIDE is not None:
        unknown = [c for c in CUTOFFS_OVERRIDE if c not in settings["cut_offs"]]
        if unknown:
            raise ValueError("GARGAML_CUTOFFS "+str(unknown)+" not in this dataset's sweep "
                             +str(settings["cut_offs"]))
        settings["cut_offs"] = CUTOFFS_OVERRIDE
    if TARGETS_OVERRIDE is not None:
        settings["targets"] = TARGETS_OVERRIDE
    return settings


def main():
    dataset = DATASET
    settings = sweep_settings(dataset)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print("epochs cap: "+str(EPOCHS)+", cut-offs: "+str(settings["cut_offs"])
          +", configs: "+str(CONFIGS)+", writing to "+OUTPUT_DIR)

    print("device: "+str(device))
    print("Transductive evaluation on the persisted folds; neighbour "
          "features are built on the full graph before folding.")

    folds_path = RESULTS_DIR+"/"+dataset+"_folds.csv"
    if not os.path.exists(folds_path):
        raise FileNotFoundError(
            folds_path+" not found -- run scripts/gargaml_tree.py with N_FOLDS >= 2 "
            "first; GraphSAGE reads that partition rather than re-deriving it.")
    folds_df = pd.read_csv(folds_path)

    if TARGETS_OVERRIDE is not None:
        # A target the folds file holds but not at every cut-off (too few
        # positives to fold) is fine: run_target reports that cell as a gap.
        # A target it never holds is a typo, which would otherwise surface as
        # a full sweep of "no fold partition" gaps after the cache loads.
        split_targets = sorted(folds_df["target"].unique())
        unknown = [t for t in TARGETS_OVERRIDE if t not in split_targets]
        if unknown:
            raise ValueError("GARGAML_TARGETS "+str(unknown)+" not in "+folds_path
                             +", which holds "+str(split_targets))
    print("targets: "+str(settings["targets"]))

    laundering_combined = load_labels(dataset)

    all_records = []
    eval_mask = None  # built once the node order is known, below
    for config in CONFIGS:
        suffix = CONFIG_SUFFIXES[config]
        print("\n=== "+dataset+", "+pretty_config(MODEL_KEY, config)+" ===")

        # Checked before the build: once it has run, the caches exist either way.
        from_cache = all(os.path.exists(p) for p in graph_cache_paths(dataset, config, RESULTS_DIR))
        data, node_order, preprocess_seconds = build_graph_data(dataset, config=config,
                                                                results_dir=RESULTS_DIR)
        print("  "+str(len(node_order))+" nodes, "+str(data.num_edges)+" directed entries, "
              +str(data.num_node_features)+" features, "
              +("loaded from cache" if from_cache else "built")+" in "
              +format(preprocess_seconds, ".1f")+" s")
        if from_cache:
            print("  preprocess_seconds is a cache load here; a prepared cold build "
                  "time is in "+prep_path(dataset))

        # Both configs share one node order (same structure, different
        # features), so this is computed once and reused.
        if eval_mask is None:
            eval_mask = evaluable_mask(laundering_combined, node_order)
            if not eval_mask.all():
                print("  bank view: "+str(int(eval_mask.sum()))+" of "+str(len(node_order))
                      +" nodes are clients and scored; the rest carry messages only")

        schema_path = OUTPUT_DIR+"/"+dataset+"_undirected"+suffix+"_feature_schema.csv"
        feature_schema(config).to_csv(schema_path, index=False)
        print("  feature schema -> "+schema_path)

        # Preprocessing is per (dataset, config), not per fold, so it is one
        # row rather than a value repeated across every fold as if it had been
        # paid each time. Kept out of the records that reach write_metrics:
        # that function takes the matrices' row and column order from every
        # row it is given, so a cutoff=NaN / target="(all)" row would add a
        # spurious row and column to each matrix file.
        preprocess_records = metric_records(
            {"preprocess_seconds": preprocess_seconds,
             "preprocess_from_cache": float(from_cache),
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

        write_metrics(records, dataset, "undirected", suffix=suffix,
                     results_dir=OUTPUT_DIR, write_std=True)
        all_records += preprocess_records + records

    write_outputs(dataset, metrics_frame(all_records))


def write_outputs(dataset, long_df):
    """The tidy, per-run and fold-summary files, both configs together."""
    # The complete record, including the per-config preprocessing rows. Those
    # carry cutoff=NaN and fold=NaN, which pivot_table drops as index keys and
    # aggregate_folds filters out, so without this file the preprocessing
    # timings would exist in no output at all.
    tidy_path = OUTPUT_DIR+"/"+dataset+"_graphsage_tidy.csv"
    long_df.to_csv(tidy_path, index=False)
    print("\ntidy metrics  -> "+tidy_path)

    runs_path = OUTPUT_DIR+"/"+dataset+"_graphsage_runs.csv"
    wide_runs_frame(long_df).to_csv(runs_path, index=False)
    print("per-run table -> "+runs_path)

    summary_path = OUTPUT_DIR+"/"+dataset+"_graphsage_summary.csv"
    aggregate_folds(long_df).to_csv(summary_path, index=False)
    print("fold summary  -> "+summary_path)


def merge(dataset):
    """Combine per-target runs into the outputs one full sweep would write.

    Reads every ``MERGE_FROM/*/<dataset>_graphsage_tidy.csv`` -- one per
    ``GARGAML_TARGETS`` job -- and writes the per-config metrics, matrices and
    schema plus the tidy, runs and summary files to ``OUTPUT_DIR``. Refuses
    to write if a (config, target) of the sweep is missing or was written by
    two parts, so a failed or repeated job cannot pass for a complete sweep.
    The sweep is resolved as for a fit, so ``GARGAML_CUTOFFS`` /
    ``GARGAML_TARGETS`` / ``GARGAML_CONFIGS`` narrow what it expects.
    """
    settings = sweep_settings(dataset)
    paths = sorted(glob.glob(os.path.join(MERGE_FROM, "*", dataset+"_graphsage_tidy.csv")))
    if not paths:
        raise FileNotFoundError("no "+dataset+"_graphsage_tidy.csv in any directory under "
                                +MERGE_FROM)
    df = pd.concat([pd.read_csv(p).assign(part=os.path.dirname(p)) for p in paths],
                   ignore_index=True)
    df = df[df["features"].isin(CONFIGS)]

    preprocess = df["target"] == "(all)"
    owners = df[~preprocess].groupby(["features", "target"])["part"].unique()
    repeated = {k: list(v) for k, v in owners.items() if len(v) > 1}
    if repeated:
        raise ValueError("written by more than one part -- remove the stale one: "
                         +str(repeated))
    missing = [(c, t) for c in CONFIGS for t in settings["targets"] if (c, t) not in owners.index]
    if missing:
        raise ValueError("missing from "+MERGE_FROM+" -- rerun these before merging: "
                         +str(missing))

    # Every part records a preprocessing row set per config; after the prep
    # job they are all cache loads (the cold builds are in prep_path), so one
    # per config is kept, as a single full sweep would have written.
    pre = df[preprocess].drop_duplicates(["features", "metric"])

    # The order a full sweep writes in: config, then cut-off, then target,
    # with each part's own fold order kept by the stable sort. write_metrics
    # takes the matrices' row and column order from it.
    rank = lambda values: {v: i for i, v in enumerate(values)}
    fits = df[~preprocess].assign(
        _c=lambda d: d["features"].map(rank(CONFIGS)),
        _k=lambda d: d["cutoff"].map(rank(settings["cut_offs"])),
        _t=lambda d: d["target"].map(rank(settings["targets"])).fillna(len(settings["targets"])))
    fits = fits.sort_values(["_c", "_k", "_t"], kind="stable").drop(columns=["_c", "_k", "_t"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []
    for config in CONFIGS:
        suffix = CONFIG_SUFFIXES[config]
        print("\n=== "+dataset+", "+pretty_config(MODEL_KEY, config)+" ===")
        rows = fits[fits["features"] == config].drop(columns="part")
        # The preprocessing rows' NaN fold made these float on the way in; a
        # fit-only frame holds integers, as a single run writes them.
        for column in ("fold", "n_test", "n_pos"):
            if rows[column].notna().all():
                rows[column] = rows[column].astype("int64")
        for target in settings["targets"]:
            print("  "+target+" <- "+owners[(config, target)][0])
        feature_schema(config).to_csv(
            OUTPUT_DIR+"/"+dataset+"_undirected"+suffix+"_feature_schema.csv", index=False)
        write_metrics(rows.to_dict("records"), dataset, "undirected", suffix=suffix,
                      results_dir=OUTPUT_DIR, write_std=True)
        frames += [pre[pre["features"] == config].drop(columns="part"), rows]

    write_outputs(dataset, metrics_frame(pd.concat(frames, ignore_index=True).to_dict("records")))


def prep_path(dataset):
    """Where ``prepare`` records what each cache build cost."""
    return RESULTS_DIR+"/"+dataset+"_graphsage_prep.csv"


def prepare(dataset):
    """Build every cache the fits read, record what each step cost, and stop.

    Labels, structure and features come from data/ alone, so this needs no
    GPU and no folds file and can run before the tree job has finished. A
    step whose cache already existed is a load, and is flagged ``from_cache``.
    """
    rows = []

    from_cache = os.path.exists(labels_cache_path(dataset))
    started = time.perf_counter()
    laundering_combined = load_labels(dataset)
    rows.append(dict(dataset=dataset, step="labels", from_cache=from_cache,
                     seconds=time.perf_counter() - started,
                     peak_host_mb=peak_host_memory_mb()))
    print("  "+str(len(laundering_combined))+" labelled accounts, "
          +("loaded" if from_cache else "built")+" in "
          +format(rows[-1]["seconds"], ".1f")+" s")
    del laundering_combined

    for config in CONFIGS:
        from_cache = all(os.path.exists(p) for p in graph_cache_paths(dataset, config, RESULTS_DIR))
        data, node_order, seconds = build_graph_data(dataset, config=config,
                                                     results_dir=RESULTS_DIR)
        print("  "+config+": "+str(len(node_order))+" nodes, "+str(data.num_edges)
              +" directed entries, "+str(data.num_node_features)+" features, "
              +("loaded" if from_cache else "built")+" in "+format(seconds, ".1f")+" s")
        rows.append(dict(dataset=dataset, step=config, from_cache=from_cache,
                         seconds=seconds, peak_host_mb=peak_host_memory_mb()))
        del data, node_order

    # peak_host_mb is the process's high-water mark so far, so it accumulates
    # down the rows: the labels row is the label build's own peak.
    write_csv(pd.DataFrame(rows), prep_path(dataset))
    print("\nprep record -> "+prep_path(dataset))


if __name__ == "__main__":
    echo_config(__file__, dataset=DATASET, configs=CONFIGS, prepare_only=PREPARE_ONLY,
                checkpoint_dir=CHECKPOINT_DIR, epochs=EPOCHS, patience=PATIENCE,
                results_dir=RESULTS_DIR, output_dir=OUTPUT_DIR, targets=TARGETS_OVERRIDE,
                merge_from=MERGE_FROM)
    if PREPARE_ONLY:
        prepare(DATASET)
    elif MERGE_FROM is not None:
        merge(DATASET)
    else:
        main()
