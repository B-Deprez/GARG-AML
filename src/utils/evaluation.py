"""
Shared evaluation for GARG-AML results (task 2).

Single code path for every model in the repository -- the tree/boosting
models on IBM data, the block-only ablation, the synthetic runs, the
isolation forest and the raw GARG-AML scores. Before this module each
script defined its own ``evaluate_model``, which is how the
``clf.predict()`` defect below survived in five places at once.

Three jobs:

1. **Continuous scores.** :func:`model_scores` returns a ranking score for
   any of the estimators used here (``predict_proba`` -> ``decision_function``
   -> ``-score_samples`` for the isolation forest). AUC-ROC and AUC-PR were
   previously computed from ``clf.predict()``, i.e. from hard 0/1 labels;
   that understates both, and the ranking metrics below need a score that
   actually orders the accounts.

2. **Ranking metrics** (reviewers R1-5 / R2-M4): Precision@K, Recall@K,
   lift@K and #TP@K, swept over realistic alert-queue sizes
   (:data:`ALERT_SIZES`) rather than fractions of the node set -- the point
   is investigator workload. Average precision is *not* added here: the
   existing ``AUC_PR`` column already is average precision, it was simply
   computed from labels instead of scores.

3. **Result writing.** :func:`write_metrics` emits one tidy CSV holding
   every metric, and reproduces the historical
   ``<dataset>_<metric>_<model>_<direction>_combined.csv`` matrices
   unchanged, so the visualisation notebooks keep working untouched.

Evaluation population
---------------------
Metrics are computed on the 30% test split only, consistent with the
precision/F1/AUC numbers in the paper. ``K`` is an absolute alert count,
so it can exceed the number of test rows or the number of positives; the
metric is still written, with ``n_test`` and ``n_pos`` beside it, so that
e.g. ``R@1000 = 1.0`` off 12 positives is readable as trivial rather than
impressive.

Ties
----
Decision trees emit few distinct probabilities, so many accounts share a
score and "the top 50" is not always uniquely defined. The metrics take
exactly K rows after a stable sort, which is the standard definition and
is reproducible; ``ties@K`` reports the size of the score group straddling
the cut, and equals 1 when the top-K set is unambiguous. Report it rather
than let an arbitrary ordering pass silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Realistic alert-queue sizes: what a team can actually work through,
# not a fraction of the node set.
ALERT_SIZES = [50, 100, 500, 1000]

# The reproducibility seed used throughout the repository.
SEED = 1997

# Metrics that already had a per-metric result file before this module
# existed. Their file names must not change: the visualisation notebooks
# glob for them.
LEGACY_METRICS = ["precision", "f1", "AUC_ROC", "AUC_PR"]

# Column order of the tidy result frame.
RECORD_COLUMNS = [
    "dataset", "direction", "model", "features",
    "cutoff", "target", "seed",
    "metric", "K", "value",
    "n_test", "n_pos", "status",
]

# Historical file-name token per canonical model key from src/utils/naming.py.
# The keys say "boost" but the result files have always said "boosting";
# the files win, because renaming them would break every notebook.
#
# Deliberately an allow-list, not a fallback-to-key dict: a model key with no
# entry here has no legacy file a notebook reads, so write_metric_matrices
# raises rather than writing an orphan CSV under its raw key. Add an entry
# (and confirm the token against what VisualisationResults.ipynb actually
# opens) as each further model is retrofitted onto this module -- base
# GARG-AML scores, the isolation forest, the block-only ablation.
LEGACY_MODEL_TOKENS = {
    "gargaml_tree_u": "tree",
    "gargaml_tree_d": "tree",
    "gargaml_boost_u": "boosting",
    "gargaml_boost_d": "boosting",
}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def holdout_split(X, y, test_size=0.3, seed=SEED):
    """The one 70/30 stratified split, so every model sees the same slice.

    Transductive by design (the split is on the feature table, not on the
    graph); task 7 varies ``seed`` for repeated splits.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def is_outlier_detector(clf):
    """True for sklearn outlier/density estimators such as IsolationForest.

    They score anomalies *low*, the opposite of every classifier here, so
    their scores have to be negated before ranking. Checked two ways
    because ``_estimator_type`` is on its way out of sklearn.
    """
    if getattr(clf, "_estimator_type", None) == "outlier_detector":
        return True
    return hasattr(clf, "score_samples") and not hasattr(clf, "predict_proba")


def model_scores(clf, X):
    """Continuous ranking score for ``X``; higher means more suspicious.

    Covers the three estimator interfaces used in this repo: ``predict_proba``
    (tree, boosting), ``decision_function``, and ``score_samples``
    (isolation forest).

    Outlier detectors are handled *before* ``decision_function``, because
    ``IsolationForest`` exposes both and its ``decision_function`` is high
    for normal points -- taking it would silently rank the most ordinary
    accounts first. ``gargaml_IF.py`` already negates by hand for the same
    reason.
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
    population -- the same *ratio* as ``lift_curve_values`` in
    scripts/distribution_scores.py, but a different *population*: that
    function takes the tie-inclusive superset at the cut value
    (``Pred >= data_value``), while this one takes exactly the top ``K``
    rows after a stable sort. On the tied plateaus a decision tree
    produces, the two numbers diverge substantially (verified: >10x on a
    6-leaf tree score), so they are not interchangeable. Exactly-``K`` is
    the right population *here*: the point is sizing a fixed-size alert
    queue (see module docstring), and ``lift_curve_values`` is answering a
    different question (a continuous curve over fractions). ``ties@K``
    flags when the top-``K`` set was arbitrary within a tie, which is
    exactly when the two definitions would disagree most. Recall@K and
    lift@K are NaN when there are no positives.
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

    When there is no natural 0/1 prediction -- the raw GARG-AML score
    (range [-1, 1]), an isolation forest's ``-score_samples``
    (~[0.35, 0.60], no operating point chosen at 0.5) -- ``precision`` and
    ``f1`` are reported as NaN rather than invented at an arbitrary 0.5
    cut: thresholding those scales at 0.5 would silently fabricate a
    number and write it into the same published column as a real
    ``clf.predict()`` result.
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

    Returns a **dict** keyed by metric name -- not the 4-tuple the scripts
    used to unpack, so that adding a metric never touches a call site.
    """
    y_score = model_scores(clf, X_test)
    y_pred = clf.predict(X_test)
    return evaluate_scores(y_test, y_score, y_pred=y_pred, alert_sizes=alert_sizes)


def nan_metrics(alert_sizes=ALERT_SIZES):
    """Metric dict of NaNs for a cell that could not be evaluated.

    Cells with too few positives have always been reported as gaps rather
    than dropped; this keeps that behaviour explicit instead of relying on
    a zero-filled matrix.
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
    """Tidy DataFrame from the record list, with a stable column order."""
    df = pd.DataFrame(records)
    columns = [c for c in RECORD_COLUMNS if c in df.columns]
    return df[columns + [c for c in df.columns if c not in columns]]


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------

def _matrix(df, index_order, column_order):
    """Pivot one metric into the historical cut-off x pattern matrix."""
    matrix = df.pivot_table(
        index="cutoff", columns="target", values="value", dropna=False
    )
    matrix = matrix.reindex(index=index_order, columns=column_order)
    matrix.index.name = None
    matrix.columns.name = None
    return matrix


def write_metric_matrices(long_df, dataset, str_directed, suffix="",
                          results_dir="results"):
    """Reproduce the historical per-metric result files from tidy records.

    Emits ``<dataset>_<metric>_<model>_<direction><suffix>_combined.csv``
    for each of :data:`LEGACY_METRICS` and each model, plus the
    model-independent ``<dataset>_imbalance_<direction><suffix>_combined.csv``.
    Layout, ordering and NaN gaps match what the pipeline wrote before, so
    notebooks reading these files need no changes.
    """
    index_order = list(pd.unique(long_df["cutoff"]))
    column_order = list(pd.unique(long_df["target"]))

    written = []

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
            _matrix(rows[rows["model"] == model], index_order, column_order).to_csv(path)
            written.append(path)

    imbalance = long_df[long_df["metric"] == "imbalance"]
    if len(imbalance):
        path = (f"{results_dir}/{dataset}_imbalance_"
                f"{str_directed}{suffix}_combined.csv")
        _matrix(imbalance, index_order, column_order).to_csv(path)
        written.append(path)

    return written


def write_metrics(records, dataset, str_directed, suffix="", results_dir="results"):
    """Write the tidy CSV *and* the historical matrices. Returns the frame.

    One call at the end of a script replaces the block of per-matrix
    ``to_csv`` lines each of them used to carry.
    """
    long_df = metrics_frame(records)

    tidy_path = f"{results_dir}/{dataset}_{str_directed}{suffix}_metrics.csv"
    long_df.to_csv(tidy_path, index=False)
    print(f"  tidy metrics -> {tidy_path}")

    written = write_metric_matrices(
        long_df, dataset, str_directed, suffix=suffix, results_dir=results_dir
    )
    print(f"  wrote {len(written)} result CSVs in the historical format")

    return long_df
