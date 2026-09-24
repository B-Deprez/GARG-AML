"""
Shared evaluation for GARG-AML results.

One code path for every model in the repository: the tree/boosting models on
IBM data, the feature-group ablations, the synthetic runs, the isolation
forest and the raw GARG-AML scores.

Three jobs:

1. **Continuous scores.** :func:`model_scores` returns a ranking score for
   any of the estimators used here (``predict_proba`` -> ``decision_function``
   -> ``-score_samples`` for the isolation forest). Every threshold-free
   metric is computed from those scores, never from hard 0/1 labels.

2. **Ranking metrics.** Precision@K, Recall@K, lift@K and #TP@K, swept over
   realistic alert-queue sizes (:data:`ALERT_SIZES`) rather than fractions of
   the node set, since the quantity of interest is investigator workload. The
   ``AUC_PR`` column is average precision, so there is no separate AP column.

3. **Result writing.** :func:`write_metrics` emits one tidy CSV holding every
   metric, plus the per-metric
   ``<dataset>_<metric>_<model>_<direction>_combined.csv`` matrices the
   visualisation notebooks read.

Label cut-offs
--------------
:data:`CUT_OFFS` is the label sweep every script reads, and
:data:`HEADLINE_CUTOFFS` the slice the manuscript reports and the expensive
runs restrict themselves to.

Evaluation population
---------------------
:func:`holdout_split` is a single 70/30 stratified split, used by the
synthetic scripts. :func:`cv_splits` is 5-fold stratified CV for the IBM
tree/boosting runs; its disjoint test folds also allow a pooled out-of-fold
pass over every account rather than a 20-30% slice. ``K`` is an absolute
alert count, so it can exceed the number of test rows or the number of
positives; the metric is still written, with ``n_test`` and ``n_pos`` beside
it, so that e.g. ``R@1000 = 1.0`` off 12 positives is readable as trivial
rather than impressive.

Folds
-----
:func:`cv_splits` returns each fold's train/test indices; :func:`write_folds`
persists the account/cutoff/target/fold membership to
``results/<dataset>_folds.csv`` so a later model reads the same partition
instead of re-deriving it. In the tidy CSV, ``fold`` is ``0..n_splits-1`` for
a per-fold row, ``-1`` for a pooled out-of-fold row, and ``NaN`` for a
single-split run. :func:`aggregate_folds` reduces the per-fold rows to
mean/std/min/max plus ``n_folds_ok``.

Ties
----
Decision trees emit few distinct probabilities, so many accounts share a
score and "the top 50" is not always uniquely defined. The metrics take
exactly K rows after a stable sort; ``ties@K`` reports the size of the score
group straddling the cut, and equals 1 when the top-K set is unambiguous.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# Realistic alert-queue sizes: what a team can work through, not a fraction of
# the node set. K=10 matters on the synthetic grid, whose 100-node datasets
# leave a 30-row test split -- every larger K collapses onto k_eff = min(K, n)
# there, so P@50 and P@100 report nothing but overall precision.
ALERT_SIZES = [10, 50, 100, 500, 1000]

# The label cut-off sweep, canonical for the whole repository.
#
# An account's label for a target column is its *propensity*: that account's
# laundering-flagged transactions of that type over all of its transactions
# (src/data/pattern_construction.py::summarise_ML_labels). Every call site
# labels with the strict comparison ``propensity > cutoff``, so 0.0 means
# "involved in at least one laundering transaction" -- the most inclusive
# labelling available, and the cut-off with the most positives, hence the one
# most likely to be evaluable where 0.5 and 0.9 come back NaN.
CUT_OFFS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.9]

# The slice reported in Tables 10-11, and the reduced sweep the expensive runs
# use -- LI-Large's entry in gargaml_tree.py's DATASET_SETTINGS and
# graphsage_baseline.py's DATASETS both point here, so the two stay comparable
# cell for cell.
HEADLINE_CUTOFFS = [0.0, 0.1, 0.5, 0.9]

# The reproducibility seed used throughout the repository.
SEED = 1997

# Default n_splits for cv_splits. Distinct from a caller's own "is CV on"
# switch -- e.g. gargaml_tree.py's N_FOLDS, where 0 means holdout_split.
CV_FOLDS = 5

# Metrics that have a per-metric result file of their own; the visualisation
# notebooks glob for those file names.
LEGACY_METRICS = ["precision", "f1", "AUC_ROC", "AUC_PR"]

# Column order of the tidy result frame. ``fold`` is 0..n_splits-1 for a
# per-fold CV row, -1 for a pooled out-of-fold row, and NaN for a single-split
# run: metric_records builds each row from a context dict, so a caller that
# passes no ``fold`` yields NaN once a DataFrame is built from mixed records.
RECORD_COLUMNS = [
    "dataset", "direction", "model", "features",
    "cutoff", "target", "seed", "fold",
    "metric", "K", "value",
    "n_test", "n_pos", "status",
]

# File-name token per canonical model key from src/utils/naming.py. The keys
# say "boost" where the result files say "boosting"; the files win, because
# renaming them would break every notebook.
#
# An allow-list, not a fallback-to-key dict: a model key with no entry raises
# in write_metric_matrices rather than writing an orphan CSV under its raw key.
LEGACY_MODEL_TOKENS = {
    "gargaml_tree_u": "tree",
    "gargaml_tree_d": "tree",
    "gargaml_boost_u": "boosting",
    "gargaml_boost_d": "boosting",
    "gargaml_if_d": "isolationforest",
    "graphsage_u": "graphsage",
    # The base GARG-AML scores (scripts/distribution_scores.py) are absent by
    # design: that script writes its tidy frame with write_matrices=False and
    # has no matrix format to reproduce.
}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def holdout_split(X, y, test_size=0.3, seed=SEED):
    """The single 70/30 stratified split, so every model sees the same slice.

    Transductive by design: the split is on the feature table, not on the
    graph. ``seed`` is fixed rather than varied over repeated random draws,
    whose overlapping test sets deflate the reported spread. Used by the
    synthetic scripts; the IBM runs use :func:`cv_splits`.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def cv_splits(X, y, n_splits=CV_FOLDS, seed=SEED):
    """5-fold (default) stratified CV splits, disjoint and exhaustive over ``y``.

    Returns a **list** of ``(fold, train_idx, test_idx)`` -- eager, not a
    generator, so a too-small class raises at the call site the way
    ``holdout_split``'s stratification failure does. The minimum class count
    is checked explicitly rather than left to ``StratifiedKFold``, which in
    some sklearn versions only warns and returns folds with a class missing.

    The ``n_splits`` test sets are disjoint and partition ``y`` exactly once
    each, which is the property a variance estimate needs and repeated random
    splits, whose test sets overlap by design, do not have. ``seed`` is the
    ``StratifiedKFold`` shuffle seed, so the partition is reproducible.
    """
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if counts.min() < n_splits:
        raise ValueError(
            f"The least populated class has {counts.min()} member(s), fewer than "
            f"n_splits={n_splits}; cannot stratify into that many folds."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y))
    ]


# ---------------------------------------------------------------------------
# Fold persistence
# ---------------------------------------------------------------------------

# results/<dataset>_folds.csv schema: one row per account that landed in a
# test fold for a given (cutoff, target). A pair with too few positives for
# cv_splits has no rows here, which means "not CV-partitioned" rather than a
# fold of 0.
FOLDS_COLUMNS = ["account", "cutoff", "target", "fold"]


def write_folds(records, dataset, results_dir="results"):
    """Persist the fold partition, so later models read it instead of
    re-deriving it."""
    path = f"{results_dir}/{dataset}_folds.csv"
    pd.DataFrame(records, columns=FOLDS_COLUMNS).to_csv(path, index=False)
    return path


def folds_path(dataset, results_dir="results"):
    """Where :func:`write_folds` put ``dataset``'s partition."""
    return f"{results_dir}/{dataset}_folds.csv"


def read_folds(dataset, results_dir="results"):
    """The persisted partition for ``dataset``, or ``None`` if there is none.

    ``None`` rather than an exception: CV is scoped to the IBM data, so the
    synthetic datasets are never partitioned, and an IBM run made with
    ``N_FOLDS = 0`` writes no partition either. In both cases the caller
    evaluates the full population instead. Callers that cannot proceed
    without folds (scripts/graphsage_baseline.py) check for ``None``
    themselves.
    """
    path = folds_path(dataset, results_dir)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def fold_assignments(folds_df, cutoff, target):
    """``{fold: [account, ...]}`` for one (cut-off, target) cell.

    Empty when the partition has no rows for that pair, which means "not
    CV-partitioned" rather than "a fold of zero accounts".
    """
    if folds_df is None:
        return {}
    rows = folds_df[(folds_df["cutoff"] == cutoff) & (folds_df["target"] == target)]
    return {int(f): group["account"].tolist() for f, group in rows.groupby("fold")}


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def is_outlier_detector(clf):
    """True for sklearn outlier/density estimators such as IsolationForest.

    They score anomalies *low*, the opposite of every classifier here, so
    their scores are negated before ranking. Checked two ways, since
    ``_estimator_type`` is deprecated in recent sklearn.
    """
    if getattr(clf, "_estimator_type", None) == "outlier_detector":
        return True
    return hasattr(clf, "score_samples") and not hasattr(clf, "predict_proba")


def model_scores(clf, X):
    """Continuous ranking score for ``X``; higher means more suspicious.

    Covers the three estimator interfaces used here: ``predict_proba`` (tree,
    boosting), ``decision_function``, and ``score_samples`` (isolation
    forest). Outlier detectors are matched *before* ``decision_function``,
    because ``IsolationForest`` exposes both and its ``decision_function`` is
    high for normal points, which would rank the most ordinary accounts first.
    """
    if hasattr(clf, "predict_proba") and not is_outlier_detector(clf):
        proba = clf.predict_proba(X)
        if proba.shape[1] == 1:
            # Only one class present at fit time: no ranking information.
            # Return that class so the caller fails on the metric, with a
            # readable message, rather than on an index error here.
            return np.full(proba.shape[0], float(clf.classes_[0]))
        return proba[:, 1]

    if is_outlier_detector(clf):
        if hasattr(clf, "score_samples"):
            return -clf.score_samples(X)
        return -clf.decision_function(X)

    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)

    if hasattr(clf, "score_samples"):
        return -clf.score_samples(X)

    raise TypeError(
        f"{type(clf).__name__} exposes none of predict_proba, "
        "decision_function or score_samples, so it cannot be ranked."
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_names(alert_sizes=ALERT_SIZES):
    """Every metric key produced by :func:`evaluate_scores`, in order."""
    names = list(LEGACY_METRICS)
    for k in alert_sizes:
        names += [f"P@{k}", f"R@{k}", f"lift@{k}", f"TP@{k}", f"ties@{k}"]
    return names


def tie_group_size(scores_sorted, k):
    """Size of the score group straddling the top-``k`` cut; 1 if clean.

    ``scores_sorted`` is in descending order. A value above 1 means the
    top-``k`` set is not uniquely determined by the scores.
    """
    n = len(scores_sorted)
    if k <= 0 or k >= n:
        return 1
    cut = scores_sorted[k - 1]
    if scores_sorted[k] != cut:
        return 1  # the tied group, if any, fits entirely inside the top k
    return int(np.sum(scores_sorted == cut))


def ranking_metrics(y_true, y_score, alert_sizes=ALERT_SIZES):
    """Precision@K, Recall@K, lift@K, #TP@K and the tie diagnostic.

    ``lift@K`` is Precision@K over the prevalence of the evaluated
    population. It is the same *ratio* as ``lift_curve_values`` in
    scripts/distribution_scores.py but over a different *population*: that
    function takes the tie-inclusive superset at the cut value
    (``Pred >= data_value``), this one exactly the top ``K`` rows after a
    stable sort, and on the tied plateaus a decision tree produces the two
    diverge substantially. Exactly-``K`` is the population for sizing a
    fixed alert queue; ``ties@K`` flags when the top-``K`` set was arbitrary
    within a tie. Recall@K and lift@K are NaN when there are no positives.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    n = len(y_true)
    n_pos = int(y_true.sum())
    prevalence = n_pos / n if n else np.nan

    # Stable sort so ties keep the input order and runs are reproducible.
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    metrics = {}
    for k in alert_sizes:
        k_eff = min(k, n)
        tp = int(y_sorted[:k_eff].sum())
        precision_at_k = tp / k_eff if k_eff else np.nan

        metrics[f"P@{k}"] = precision_at_k
        metrics[f"R@{k}"] = tp / n_pos if n_pos else np.nan
        metrics[f"lift@{k}"] = precision_at_k / prevalence if n_pos else np.nan
        metrics[f"TP@{k}"] = tp
        metrics[f"ties@{k}"] = tie_group_size(scores_sorted, k_eff)

    return metrics


def evaluate_scores(y_true, y_score, y_pred=None, alert_sizes=ALERT_SIZES):
    """All metrics for one (model, cut-off, pattern) cell, as a flat dict.

    ``y_score`` drives the threshold-free metrics; ``y_pred`` drives
    precision and F1. Pass the model's own ``predict`` output as
    ``y_pred`` when one exists (:func:`evaluate_model` always does).

    When there is no natural 0/1 prediction -- the raw GARG-AML score (range
    [-1, 1]), an isolation forest's ``-score_samples`` -- ``precision`` and
    ``f1`` are NaN rather than invented at an arbitrary 0.5 cut, which would
    write a fabricated number into the same column as a real ``predict``
    result.
    """
    if y_pred is None:
        precision = f1 = np.nan
    else:
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

    metrics = {
        "precision": precision,
        "f1": f1,
        "AUC_ROC": roc_auc_score(y_true, y_score),
        "AUC_PR": average_precision_score(y_true, y_score),
    }
    metrics.update(ranking_metrics(y_true, y_score, alert_sizes=alert_sizes))
    return metrics


def evaluate_model(clf, X_test, y_test, alert_sizes=ALERT_SIZES):
    """Fit-free evaluation of ``clf`` on the test split.

    Returns a dict keyed by metric name, so adding a metric never touches a
    call site.
    """
    y_score = model_scores(clf, X_test)
    y_pred = clf.predict(X_test)
    return evaluate_scores(y_test, y_score, y_pred=y_pred, alert_sizes=alert_sizes)


def nan_metrics(alert_sizes=ALERT_SIZES):
    """Metric dict of NaNs for a cell that could not be evaluated.

    A cell with too few positives to fit is reported as a gap rather than
    dropped or zero-filled.
    """
    return {name: np.nan for name in metric_names(alert_sizes)}


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

def metric_records(metrics, status="ok", n_test=np.nan, n_pos=np.nan, **context):
    """Turn a metric dict into tidy rows.

    ``context`` supplies the identifying columns (dataset, direction,
    model, features, cutoff, target, seed). Keys of the form ``"P@100"``
    are split into ``metric="P@K"`` and ``K=100`` so a K sweep can be
    filtered without string parsing downstream.
    """
    records = []
    for name, value in metrics.items():
        if "@" in name:
            metric, k = name.split("@")
            metric, k = metric + "@K", int(k)
        else:
            metric, k = name, np.nan

        row = dict(context)
        row.update(
            metric=metric, K=k, value=value,
            n_test=n_test, n_pos=n_pos, status=status,
        )
        records.append(row)

    return records


def metrics_frame(records):
    """Tidy DataFrame from the record list, with a stable column order.

    ``fold`` is guaranteed to exist even when no record in the batch passes
    one: pandas creates a NaN column only for rows missing a key some other
    row supplied, so a batch from a single-split caller would otherwise have
    no such column and callers that assume one (write_metric_matrices's
    ``fold == -1`` check) would raise ``KeyError``.
    """
    df = pd.DataFrame(records)
    if "fold" not in df.columns:
        df["fold"] = np.nan
    columns = [c for c in RECORD_COLUMNS if c in df.columns]
    return df[columns + [c for c in df.columns if c not in columns]]


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------

def _matrix(df, index_order, column_order, aggfunc="mean"):
    """Pivot one metric into the cut-off x pattern matrix.

    ``aggfunc`` is "mean" for the matrix itself and "std" for its companion;
    with one row per (cutoff, target) cell, as every single-split caller has,
    both are no-ops on a single value.
    """
    matrix = df.pivot_table(
        index="cutoff", columns="target", values="value", dropna=False, aggfunc=aggfunc
    )
    matrix = matrix.reindex(index=index_order, columns=column_order)
    matrix.index.name = None
    matrix.columns.name = None
    return matrix


def write_metric_matrices(long_df, dataset, str_directed, suffix="",
                          results_dir="results", write_std=False):
    """Write the per-metric result files from tidy records.

    Emits ``<dataset>_<metric>_<model>_<direction><suffix>_combined.csv`` for
    each of :data:`LEGACY_METRICS` and each model, plus the model-independent
    ``<dataset>_imbalance_<direction><suffix>_combined.csv``. A cell the
    models could not be fitted on stays a NaN gap.

    Pooled out-of-fold rows (``fold == -1``) are always excluded before
    pivoting: averaging them in with the per-fold rows would mix a sixth,
    differently-computed value into the mean.

    ``write_std=True`` also writes ``..._std_combined.csv`` companions with
    ``aggfunc="std"``, same shape, so figures can gain error bars.
    """
    if "fold" in long_df.columns:
        long_df = long_df[long_df["fold"] != -1]

    index_order = list(pd.unique(long_df["cutoff"]))
    column_order = list(pd.unique(long_df["target"]))

    written = []

    def _write(rows, path):
        _matrix(rows, index_order, column_order).to_csv(path)
        written.append(path)
        if write_std:
            std_path = path.replace("_combined.csv", "_std_combined.csv")
            _matrix(rows, index_order, column_order, aggfunc="std").to_csv(std_path)
            written.append(std_path)

    for metric in LEGACY_METRICS:
        rows = long_df[long_df["metric"] == metric]
        for model in pd.unique(rows["model"]):
            if model not in LEGACY_MODEL_TOKENS:
                raise KeyError(
                    f"No legacy filename token for model key {model!r}. Add it to "
                    "LEGACY_MODEL_TOKENS in src/utils/evaluation.py -- a silent "
                    "fallback would write a file no notebook reads."
                )
            token = LEGACY_MODEL_TOKENS[model]
            path = (f"{results_dir}/{dataset}_{metric}_{token}_"
                    f"{str_directed}{suffix}_combined.csv")
            _write(rows[rows["model"] == model], path)

    imbalance = long_df[long_df["metric"] == "imbalance"]
    if len(imbalance):
        path = (f"{results_dir}/{dataset}_imbalance_"
                f"{str_directed}{suffix}_combined.csv")
        _write(imbalance, path)

    return written


def write_metrics(records, dataset, str_directed, suffix="", results_dir="results",
                   write_std=False, write_matrices=True):
    """Write the tidy CSV *and* the per-metric matrices. Returns the frame.

    ``write_std`` also emits the fold-std companion matrices; see
    :func:`write_metric_matrices`.

    ``write_matrices=False`` writes the tidy frame only. It is for a caller
    whose model has no matrix format to reproduce -- the base GARG-AML score
    in scripts/distribution_scores.py, whose numbers live in
    ``results_performance_IBM_<direction>.txt``. That caller also mixes
    per-fold rows with a full-population row, which ``_matrix``'s
    ``aggfunc="mean"`` would average together into one cell.
    """
    long_df = metrics_frame(records)

    tidy_path = f"{results_dir}/{dataset}_{str_directed}{suffix}_metrics.csv"
    long_df.to_csv(tidy_path, index=False)
    print(f"  tidy metrics -> {tidy_path}")

    if not write_matrices:
        return long_df

    written = write_metric_matrices(
        long_df, dataset, str_directed, suffix=suffix, results_dir=results_dir,
        write_std=write_std,
    )
    print(f"  wrote {len(written)} result CSVs in the historical format")

    return long_df


def aggregate_folds(long_df):
    """Mean/std/min/max across CV folds, plus ``n_folds_ok``.

    Over rows with ``fold >= 0`` only, which excludes both NaN (single-split)
    rows and the ``-1`` pooled out-of-fold row, so a pooled value is never
    averaged in as an extra fold. ``n_folds_ok`` counts folds with
    ``status == "ok"``, not folds present, so a cell where only 3 of 5 folds
    had enough positives to fit announces that rather than averaging 3
    numbers as if nothing were missing.
    """
    keys = ["dataset", "direction", "model", "features", "cutoff", "target", "metric", "K"]
    per_fold = long_df[long_df["fold"] >= 0]

    agg = per_fold.groupby(keys, dropna=False)["value"].agg(["mean", "std", "min", "max"])

    ok = per_fold[per_fold["status"] == "ok"]
    n_ok = ok.groupby(keys, dropna=False)["fold"].nunique()
    agg["n_folds_ok"] = n_ok.reindex(agg.index).fillna(0).astype(int)

    return agg.reset_index()
