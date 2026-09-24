"""
Tidy metrics -> the tables the manuscript reports.

The models write their numbers to
``results/<dataset>_<direction><suffix>_metrics.csv``; this module assembles
those rows into the results, ablation, alert-queue, sweep, cost and variance
tables, as LaTeX and as CSV.

Why here and not in ``VisualisationResults.ipynb``
--------------------------------------------------
That notebook builds exact filenames for eight hard-coded models and the four
per-metric matrices, and parses the base scores out of a free-text log with
``eval()``. The tidy CSV already carries every column these tables need, so
they are assembled from it in code that can be tested.

The one thing to know about the schema
--------------------------------------
``fold`` distinguishes three kinds of row, and a table that mixes them is
wrong in a way that is hard to see:

* ``fold >= 0`` -- one cross-validation fold. Mean and std over these is the
  variance estimate.
* ``fold == -1`` -- the pooled out-of-fold pass: every account scored by a
  model that never trained on it. This is the **only** correct population for
  the alert-queue metrics, because P@1000 against a 20 % test slice is a
  rescaled proxy rather than an investigator's queue.
* ``fold`` NaN -- a single-split run (the synthetic scripts, or any IBM run
  made with ``N_FOLDS = 0``), and the full-population row in
  ``distribution_scores.py``.

:func:`summarise` picks one kind deliberately and says which; it never
averages across them.

Gaps are reported, not hidden
-----------------------------
A cell whose models could not be fitted carries ``status`` starting with
``"skipped:"`` and a NaN value; :func:`format_cell` renders those as ``--``
rather than dropping the row, and a mean over fewer folds than expected is
marked so a reader cannot mistake it for a complete one. :func:`coverage`
reports what is on disk before any table is built.
"""

from __future__ import annotations

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

from src.utils.evaluation import ALERT_SIZES, HEADLINE_CUTOFFS, LEGACY_METRICS
from src.utils.features import is_direction_free
from src.utils.graph_processing import DEFAULT_RESOLUTION, parse_resolution
from src.utils.naming import MODEL_ORDER, pretty_config

# Where a tidy file's name comes apart: <dataset>_<direction><suffix>_metrics.csv.
# The dataset may itself contain underscores ("HI-Small_bank012",
# "synthetic_Watts-Strogatz_100_5_0.01_3"), so the direction anchors the split
# rather than a naive rsplit.
TIDY_PATTERN = re.compile(r"^(?P<dataset>.+)_(?P<direction>undirected|directed)"
                          r"(?P<suffix>.*)_metrics\.csv$")

# scripts/partial_observability.py writes one tidy file per view covering both
# directions, so it carries a `direction` COLUMN instead of a filename token.
# A direction-less name has no token to anchor the split the way TIDY_PATTERN's
# does, and the dataset can itself contain underscores ("HI-Small_bank012"), so
# the suffix is matched against this list rather than guessed by position.
NO_DIRECTION_SUFFIXES = ("partial_observability",)
NO_DIRECTION_PATTERN = re.compile(
    r"^(?P<dataset>.+)(?P<suffix>_(?:" + "|".join(NO_DIRECTION_SUFFIXES) + r"))"
    r"_metrics\.csv$")

# Metrics that are ranked at an alert-queue size rather than being a single
# number. Their rows carry a K; every other metric's K is NaN.
AT_K_METRICS = ["P@K", "R@K", "lift@K", "TP@K", "ties@K"]

# GraphSAGE's timing instrumentation writes these as metric rows too. They are
# costs, not scores, so a table of them is never bolded by maximum.
COST_METRICS = ["fit_seconds", "infer_seconds", "peak_host_mb", "peak_gpu_mb",
                "epochs_run", "epochs_to_best"]

# Tables 10-11's slice of the grid; the cut-off half is HEADLINE_CUTOFFS from
# src/utils/evaluation.py, so the reported slice and the sweep the expensive
# runs execute cannot drift apart.
HEADLINE_TARGETS = ["Is Laundering", "GATHER-SCATTER", "SCATTER-GATHER"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_name(filename):
    """Split a tidy metrics filename into dataset / direction / suffix.

    Tries the direction-anchored ``TIDY_PATTERN`` first; a direction-less
    file (see ``NO_DIRECTION_PATTERN`` above) is tried second, with
    ``direction`` coming back ``None`` -- read its ``direction`` column
    instead, per-row, rather than trusting the filename.
    """
    name = os.path.basename(filename)
    match = TIDY_PATTERN.match(name)
    if match is not None:
        return match.groupdict()
    match = NO_DIRECTION_PATTERN.match(name)
    if match is not None:
        groups = match.groupdict()
        groups["direction"] = None
        return groups
    return None


def metric_files(results_dir="results", datasets=None, exclude_suffixes=("_diagnosis",)):
    """Every tidy metrics file on disk, as ``(path, dataset, direction, suffix)``.

    ``exclude_suffixes`` drops files that are not model results. The directed
    diagnosis writes into the same tidy schema, since it reuses the shared
    metrics, but its rows are score *variants* of one model rather than
    models, so they would appear as extra rows in every results table.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.csv"))):
        parts = _parse_name(path)
        if parts is None:
            continue
        if any(parts["suffix"].endswith(s) for s in exclude_suffixes):
            continue
        if datasets is not None and parts["dataset"] not in datasets:
            continue
        found.append((path, parts["dataset"], parts["direction"], parts["suffix"]))
    return found


def load_metrics(results_dir="results", datasets=None):
    """Concatenate every tidy metrics file into one frame.

    The ``dataset`` / ``direction`` / ``features`` columns are already inside
    each file, so nothing is inferred from the filename except which files to
    read -- a file whose contents disagree with its name would be visible
    rather than silently relabelled. ``source`` records where each row came
    from, which is what makes a surprising number traceable.
    """
    frames = []
    for path, _, _, suffix in metric_files(results_dir, datasets):
        frame = pd.read_csv(path)
        if "fold" not in frame.columns:
            frame["fold"] = np.nan
        frame["suffix"] = suffix
        frame["source"] = os.path.basename(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["dataset", "direction", "model", "features",
                                     "cutoff", "target", "seed", "fold", "metric",
                                     "K", "value", "n_test", "n_pos", "status",
                                     "suffix", "source"])
    return pd.concat(frames, ignore_index=True)


def model_rows(df):
    """Rows that belong to a model.

    The ``imbalance`` metric is written once per (cut-off, pattern) cell with
    an empty ``model``, because it describes the cell rather than any
    estimator -- it reads back as NaN. Those rows are real and wanted in the
    tidy file, but they are not a model's results, so anything that groups by
    model has to drop them first or they appear as a nameless extra row.
    """
    return df[df["model"].notna() & (df["model"] != "")]


def coverage(df):
    """What is on disk, per (dataset, direction, model, features).

    Separates "this model scores badly" from "this model has no rows here".
    ``folds`` of ``single`` means the rows come from a single-split run
    rather than from cross-validation.
    """
    if df.empty:
        return df

    def _kind(group):
        if (group["fold"] >= 0).any():
            n = int(group.loc[group["fold"] >= 0, "fold"].nunique())
            return f"{n}-fold" + (" +pooled" if (group["fold"] == -1).any() else "")
        return "single"

    rows = []
    keys = ["dataset", "direction", "model", "features"]
    for key, group in model_rows(df).groupby(keys, dropna=False):
        ok = group["status"] == "ok"
        rows.append(dict(zip(keys, key), folds=_kind(group),
                         cells=group[["cutoff", "target"]].drop_duplicates().shape[0],
                         rows=len(group),
                         pct_ok=round(100 * ok.mean(), 1),
                         source=group["source"].iloc[0]))
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

GROUP_KEYS = ["dataset", "direction", "model", "features", "cutoff", "target",
              "metric", "K"]


def summarise(df, fold_mode="auto", group_keys=GROUP_KEYS):
    """Collapse to one row per cell, with ``mean``, ``std`` and ``n_folds_ok``.

    ``fold_mode`` selects which kind of row is summarised, and the three kinds
    are never mixed:

    ``"per_fold"``
        ``fold >= 0`` only -- mean +/- std across folds.
    ``"pooled"``
        ``fold == -1`` only -- the pooled out-of-fold pass. ``std`` is NaN;
        there is one value by construction.
    ``"single"``
        ``fold`` NaN only -- a single-split or full-population run.
    ``"auto"``
        per-fold where a cell has folds, single where it does not. This is
        what a table spanning both cross-validated and single-split results
        wants, and ``n_folds_ok`` tells the two apart afterwards.
    ``"pooled_first"``
        pooled where a cell has a pooled pass, single where it does not.
        What the alert-queue table wants: taking ``"pooled"`` alone would
        **drop** every model without a pooled pass, so a table meant to
        compare models would show only some of them. The populations can
        then differ between rows, which is why :func:`alert_table` prints
        ``n_test`` beside the numbers.

    ``n_folds_ok`` counts folds whose ``status`` is ``"ok"``, not folds
    present, so a mean over 3 of 5 folds announces itself.
    """
    if df.empty:
        return df.assign(mean=[], std=[], n_folds_ok=[])

    per_fold = df[df["fold"] >= 0]
    pooled = df[df["fold"] == -1]
    single = df[df["fold"].isna()]

    if fold_mode == "per_fold":
        chosen = per_fold
    elif fold_mode == "pooled":
        chosen = pooled
    elif fold_mode == "single":
        chosen = single
    elif fold_mode in ("auto", "pooled_first"):
        # One kind wins wherever it exists; ``single`` fills the rest. The
        # anti-join is on the cell keys rather than on whole rows, so a
        # dataset holding both kinds keeps both instead of losing the models
        # that have no folds.
        preferred = per_fold if fold_mode == "auto" else pooled
        cells = preferred[group_keys].drop_duplicates()
        merged = single.merge(cells.assign(_covered=1), on=group_keys, how="left")
        chosen = pd.concat([preferred, merged[merged["_covered"].isna()].drop(
            columns="_covered")], ignore_index=True)
    else:
        raise ValueError(f"Unknown fold_mode {fold_mode!r}; expected one of "
                         "'auto', 'pooled_first', 'per_fold', 'pooled', "
                         "'single'.")

    if chosen.empty:
        return pd.DataFrame(columns=group_keys + ["mean", "std", "basis",
                                                  "n_folds_ok", "n_test",
                                                  "n_pos"])

    agg = chosen.groupby(group_keys, dropna=False).agg(
        mean=("value", "mean"), std=("value", "std"),
        n_test=("n_test", "max"), n_pos=("n_pos", "max"))

    # Which kind of row this cell was built from. Needed downstream, not just
    # informative: a partial-fold marker is only meaningful for a per-fold
    # mean, and stamping "1/5" on a single-split or pooled cell would claim
    # four folds were lost when none were ever expected.
    def _basis(fold):
        if (fold >= 0).any():
            return "per_fold"
        if (fold == -1).any():
            return "pooled"
        return "single"

    agg["basis"] = chosen.groupby(group_keys, dropna=False)["fold"].apply(
        _basis).reindex(agg.index)

    ok = chosen[chosen["status"] == "ok"]
    n_ok = ok.groupby(group_keys, dropna=False)["fold"].apply(
        lambda f: int((f >= 0).sum()) if (f >= 0).any() else 1)
    agg["n_folds_ok"] = n_ok.reindex(agg.index).fillna(0).astype(int)

    return agg.reset_index()


def with_direction_free(df, direction):
    """Rows for ``direction``, plus every direction-free config's rows.

    ``topology`` is computed on the undirected reduced graph in both passes,
    so ``gargaml_tree.py`` runs it **once** and stores it under whichever
    direction ran first. Filtering on ``direction == "directed"`` alone would
    therefore drop the degree-only ablation entirely, which reads like a
    failed run rather than a shared config.
    """
    if df.empty:
        return df
    free = df["features"].map(
        lambda f: is_direction_free(f) if f in {"full", "blocks", "topology", "all"}
        else False)
    return df[(df["direction"] == direction) | free]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_cell(mean, std=np.nan, n_folds_ok=None, n_folds=None, scale=1.0,
                digits=3, bold=False, latex=True, basis="per_fold"):
    """One table cell: ``mean +/- std``, with gaps and partial folds marked.

    * a NaN mean renders as ``--`` (a cell the models could not be fitted on,
      or one outside this dataset's sweep -- both are gaps, and the tidy
      ``status`` column says which);
    * ``std`` is omitted when there is only one value, rather than printed as
      ``+/- nan``;
    * a mean over fewer folds than expected gets a trailing ``*`` and the
      count, because ``0.31 +/- 0.02`` over 3 of 5 folds is not the same
      claim as over 5. Only a per-fold ``basis`` is marked this way: a
      single-split or pooled cell has one value by construction, and marking
      it "1/5" would report four folds as missing that were never expected.
    """
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"

    text = f"{mean * scale:.{digits}f}"
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        pm = r" \pm " if latex else " +/- "
        text += pm + f"{std * scale:.{digits}f}"

    if (basis == "per_fold" and n_folds and n_folds_ok is not None
            and 0 < n_folds_ok < n_folds):
        text += f"^{{*{n_folds_ok}/{n_folds}}}" if latex else f" *{n_folds_ok}/{n_folds}"

    if latex:
        text = f"$\\mathbf{{{text}}}$" if bold else f"${text}$"
    return text


def _pivot(summary, index, columns, n_folds=None, scale=1.0, digits=3,
           bold_max=True, latex=True):
    """Pivot a summary frame into a formatted table.

    Bolding is per row (one index entry), which is how Tables 10-11 read: the
    best model for a given cut-off and pattern. Disabled for cost metrics,
    where the largest number is the worst one.
    """
    if summary.empty:
        return pd.DataFrame()

    best = (summary.groupby(index)["mean"].transform("max") if bold_max
            else pd.Series(np.nan, index=summary.index))

    summary = summary.assign(_cell=[
        format_cell(row["mean"], row["std"], row["n_folds_ok"], n_folds,
                    scale=scale, digits=digits, latex=latex,
                    basis=row.get("basis", "per_fold"),
                    bold=bold_max and np.isclose(row["mean"], b, equal_nan=False))
        for (_, row), b in zip(summary.iterrows(), best)])

    table = summary.pivot_table(index=index, columns=columns, values="_cell",
                                aggfunc="first", dropna=False)
    return table.fillna("--")


def model_label(model, features):
    """Display name for a (model, feature config) pair."""
    return pretty_config(model, features)


def _model_sort_key(model):
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

def ablation_table(df, dataset, direction, metric="AUC_PR", n_folds=None,
                   latex=True):
    """The four feature configs side by side.

    Rows are (cut-off, pattern), columns the configs. ``topology`` carries no
    GARG-AML signal at all, so the table separates how much of the lift is
    the block layout from how much is plain degree.
    """
    sub = with_direction_free(model_rows(df), direction)
    sub = sub[(sub["dataset"] == dataset) & (sub["metric"] == metric)
              & sub["model"].str.startswith("gargaml_")]
    if sub.empty:
        return pd.DataFrame()

    summary = summarise(sub)
    summary["variant"] = [
        model_label(m, f) for m, f in zip(summary["model"], summary["features"])]
    return _pivot(summary, index=["target", "cutoff"], columns="variant",
                  n_folds=n_folds, latex=latex)


def results_table(df, dataset, metric="AUC_PR", cutoffs=None, targets=None,
                  n_folds=None, scale=100.0, digits=1, latex=True):
    """Tables 10-11: every model that has rows, GraphSAGE included.

    Restricted to the headline cut-offs and patterns by default, and scaled by
    100 to match the tables in the paper. Every model appears under its
    :mod:`src.utils.naming` label, so a GraphSAGE feature config and a
    feature-group ablation cannot collide in the same column.
    """
    cutoffs = HEADLINE_CUTOFFS if cutoffs is None else cutoffs
    targets = HEADLINE_TARGETS if targets is None else targets

    sub = df[(df["dataset"] == dataset) & (df["metric"] == metric)
             & df["cutoff"].isin(cutoffs) & df["target"].isin(targets)]
    if sub.empty:
        return pd.DataFrame()

    summary = summarise(sub)
    summary["variant"] = [
        model_label(m, f) for m, f in zip(summary["model"], summary["features"])]
    summary = summary.sort_values(
        "model", key=lambda s: s.map(_model_sort_key), kind="mergesort")

    return _pivot(summary, index=["target", "cutoff"], columns="variant",
                  n_folds=n_folds, scale=scale, digits=digits, latex=latex)


def alert_table(df, dataset, cutoff, target, metric="P@K", alert_sizes=None,
                n_folds=None, latex=True, prefer_pooled=True):
    """The alert-queue table.

    One row per model, one column per queue size. Built from the **pooled
    out-of-fold** rows where they exist: the folds are disjoint, so pooling
    gives every account a score from a model that never trained on it, and K
    is then ranked over the whole dataset rather than over a 20 % slice.
    Without that, P@1000 is a rescaled proxy for an investigator's workload
    rather than the thing itself.

    The companion to :func:`ties_table`. A decision tree emits few distinct
    scores, so the top-K set can be decided by sort order inside a tied
    plateau, and ``ties@K`` is where that shows.
    """
    alert_sizes = ALERT_SIZES if alert_sizes is None else alert_sizes

    sub = df[(df["dataset"] == dataset) & (df["metric"] == metric)
             & (df["cutoff"] == cutoff) & (df["target"] == target)
             & df["K"].isin(alert_sizes)]
    if sub.empty:
        return pd.DataFrame()

    summary = summarise(sub, fold_mode="pooled_first" if prefer_pooled else "auto")
    if summary.empty:
        return pd.DataFrame()

    summary["variant"] = [
        model_label(m, f) for m, f in zip(summary["model"], summary["features"])]
    summary = summary.sort_values(
        "model", key=lambda s: s.map(_model_sort_key), kind="mergesort")

    digits = 0 if metric in ("TP@K", "ties@K") else 3
    table = _pivot(summary, index="variant", columns="K", n_folds=n_folds,
                   digits=digits, bold_max=(metric != "ties@K"), latex=latex)
    table.columns = [f"K={int(k)}" for k in table.columns]

    # Rows can come from different populations: a model with a pooled
    # out-of-fold pass is ranked over every account, a single-split one over
    # its 30 % test slice. That is a real difference in what P@K means, so
    # the population is printed rather than left to be assumed away.
    population = summary.groupby("variant")["n_test"].max()
    table.insert(0, "ranked over", population.reindex(table.index)
                 .map(lambda n: "--" if pd.isna(n) else f"{int(n):,}"))
    return table


def ties_table(df, dataset, cutoff, target, **kwargs):
    """The tie diagnostic that belongs beside every :func:`alert_table`."""
    return alert_table(df, dataset, cutoff, target, metric="ties@K", **kwargs)


def cost_table(df, dataset, metrics=None, n_folds=None, latex=True):
    """The scalability half: fit / inference time and peak memory.

    Only models that record costs appear, which is GraphSAGE. GARG-AML's
    pre-processing and scoring times are written by the measure scripts to
    ``results/time_results_*.txt`` in a different format and are not joined
    here; fit time is reported separately because GARG-AML's is zero, which
    a combined wall-clock number would hide.
    """
    metrics = COST_METRICS if metrics is None else metrics

    sub = df[(df["dataset"] == dataset) & df["metric"].isin(metrics)]
    if sub.empty:
        return pd.DataFrame()

    summary = summarise(sub)
    summary["variant"] = [
        model_label(m, f) for m, f in zip(summary["model"], summary["features"])]
    return _pivot(summary, index="variant", columns="metric", n_folds=n_folds,
                  digits=1, bold_max=False, latex=latex)


def louvain_setting(df):
    """Split ``dataset`` into its base name and its Louvain setting.

    The sweep encodes the setting in the dataset string (``HI-Small_res20``,
    ``HI-Small_nolouvain``), so every arm arrives here as a separate
    "dataset". The ``base_dataset`` and ``resolution`` columns are what let a
    table put the setting on an axis instead of scattering it across several.

    ``resolution`` is the string ``"off"`` for the no-Louvain arm rather than
    NaN: it is a real setting that produced real numbers, and a NaN would be
    dropped by the pivot and vanish from the comparison.
    """
    if df.empty:
        return df.assign(base_dataset=[], resolution=[])

    parsed = [parse_resolution(name) for name in df["dataset"]]
    return df.assign(
        base_dataset=[base for base, _ in parsed],
        resolution=["off" if res is None else res for _, res in parsed])


def _resolution_order(values):
    """Sweep columns in order of how much they reduce the graph.

    ``off`` first (nothing removed), then ascending resolution, since higher
    resolution means smaller communities and more inter-community edges
    discarded. Sorting these as strings would put ``"10"`` before ``"5"``.
    """
    numeric = sorted(v for v in values if v != "off")
    return (["off"] if "off" in set(values) else []) + numeric


def sweep_table(df, dataset, direction, metric="AUC_PR", features="full",
                cutoffs=None, targets=None, n_folds=None, latex=True):
    """Downstream performance against the Louvain setting.

    Rows are (model, pattern, cut-off); columns are the resolution, with the
    no-Louvain arm first. ``dataset`` is the **base** name (``HI-Small``),
    not one arm of the sweep -- every arm is gathered by
    :func:`louvain_setting`.

    Restricted to the ``full`` feature configuration by default: the question
    here is whether the *pre-processing* choice moves the result, so varying
    the feature groups at the same time would confound two sensitivities that
    are reported separately.
    """
    cutoffs = HEADLINE_CUTOFFS if cutoffs is None else cutoffs
    targets = HEADLINE_TARGETS if targets is None else targets

    sub = louvain_setting(model_rows(df))
    sub = sub[(sub["base_dataset"] == dataset) & (sub["direction"] == direction)
              & (sub["metric"] == metric) & (sub["features"] == features)
              & sub["cutoff"].isin(cutoffs) & sub["target"].isin(targets)]
    if sub.empty or sub["resolution"].nunique() < 2:
        return pd.DataFrame()

    summary = summarise(sub, group_keys=GROUP_KEYS + ["resolution"])
    summary["variant"] = [
        model_label(m, f) for m, f in zip(summary["model"], summary["features"])]
    summary = summary.sort_values(
        "model", key=lambda s: s.map(_model_sort_key), kind="mergesort")

    # Bolding per row would mark the best resolution for each cell, which is
    # exactly the wrong reading: the sweep is asking whether the choice
    # matters, not inviting one to be selected on the test data.
    table = _pivot(summary, index=["variant", "target", "cutoff"],
                   columns="resolution", n_folds=n_folds, bold_max=False,
                   latex=latex)
    table = table.reindex(columns=_resolution_order(table.columns))
    table.columns = [("no Louvain" if c == "off" else
                      f"r={c:g}" + (" (published)" if c == DEFAULT_RESOLUTION else ""))
                     for c in table.columns]
    return table


def severance_table(results_dir="results", latex=True):
    """The percentage of edges the pre-processing discards.

    Reads ``results/louvain_severance.csv``, which the measure scripts append
    to on every run. That file is an append-only log, so a dataset run twice
    at the same setting appears twice; the last row wins, being the one that
    produced the measures on disk.
    """
    path = os.path.join(results_dir, "louvain_severance.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    log = pd.read_csv(path)
    if log.empty:
        return pd.DataFrame()

    log["dataset"] = [parse_resolution(d)[0] for d in log["dataset"]]
    log = log.drop_duplicates(subset=["dataset", "resolution"], keep="last")

    # "off" arrives as a string and the resolutions as floats, so the column
    # is object-typed; coerce the numeric ones back for a sensible order.
    log["resolution"] = [v if v == "off" else float(v) for v in log["resolution"]]

    table = log.pivot_table(index="dataset", columns="resolution",
                            values="pct_severed", aggfunc="last")
    table = table.reindex(columns=_resolution_order(table.columns))
    # Series.map rather than DataFrame.map/applymap: the former is stable
    # across pandas versions, while DataFrame.map only exists from 2.1 and
    # applymap is deprecated from the same release.
    table = table.apply(lambda column: column.map(
        lambda v: "--" if pd.isna(v)
        else format_cell(v, digits=2, latex=latex, basis="single")))
    table.columns = [("no Louvain" if c == "off" else
                      f"r={c:g}" + (" (published)" if c == DEFAULT_RESOLUTION else ""))
                     for c in table.columns]
    table.index.name = None
    return table


# The two shapes GARG-AML is built to find, so they lead the pattern-splitting
# tables; "ALL" is the pooled row and goes last.
GARGAML_TARGETS = ["GATHER-SCATTER", "SCATTER-GATHER"]

# Column of the pattern-splitting summary -> (caption phrase, decimals).
SPLITTING_COLUMNS = {
    "pct_destroyed": ("laundering attempts left undetectable", 1),
    "mean_path_survival": ("of each attempt's two-hop paths surviving", 1),
    "pct_split": ("attempts whose accounts span more than one community", 1),
    "mean_edge_survival": ("of each attempt's own edges surviving", 1),
}


def pattern_splitting_table(results_dir="results", column="pct_destroyed",
                            latex=True):
    """What the Louvain step destroys, by pattern type.

    Rows are (dataset, pattern type), columns the resolution. Built from
    ``results/pattern_splitting_summary.csv``, which
    ``scripts/pattern_splitting.py`` writes.

    ``pct_destroyed`` is the default and the headline: the share of attempts
    that had a source-mule-target path and no longer do. It is **NaN for
    one-hop pattern types** (FAN-OUT, FAN-IN), which have no two-hop
    structure to lose, and those cells render as ``--`` rather than as a
    zero that would read like "nothing was destroyed".

    The companion to :func:`severance_table`: edges severed is what the
    pre-processing costs the graph, this is what it costs detection, and the
    two do not move together.
    """
    if column not in SPLITTING_COLUMNS:
        raise KeyError(f"Unknown splitting column {column!r}; expected one of "
                       f"{sorted(SPLITTING_COLUMNS)}.")

    path = os.path.join(results_dir, "pattern_splitting_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    summary = pd.read_csv(path)
    if summary.empty or column not in summary.columns:
        return pd.DataFrame()

    summary["resolution"] = [v if v == "off" else float(v)
                             for v in summary["resolution"]]

    table = summary.pivot_table(index=["dataset", "pattern_type"],
                                columns="resolution", values=column,
                                aggfunc="last", dropna=False)
    table = table.reindex(columns=_resolution_order(table.columns))

    # Targets first, then the other types alphabetically, then the pooled row.
    types = [t for t in table.index.get_level_values(1).unique()]
    order = (GARGAML_TARGETS
             + sorted(t for t in types if t not in GARGAML_TARGETS and t != "ALL")
             + (["ALL"] if "ALL" in types else []))
    table = table.reindex(index=[t for t in order if t in types], level=1)

    table = table.apply(lambda col: col.map(
        lambda v: "--" if pd.isna(v)
        else format_cell(v, digits=SPLITTING_COLUMNS[column][1], latex=latex,
                         basis="single")))
    table.columns = [("no Louvain" if c == "off" else
                      f"r={c:g}" + (" (published)" if c == DEFAULT_RESOLUTION else ""))
                     for c in table.columns]
    return table


def variance_table(df, dataset, metric="AUC_PR", n_folds=None, latex=True):
    """The fold spread, and how many folds each mean rests on.

    The companion to every other table here: a mean is only interpretable
    beside its spread, and a spread only beside the number of folds that
    produced it.
    """
    sub = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
    summary = summarise(sub, fold_mode="per_fold")
    if summary.empty:
        return pd.DataFrame()

    summary["variant"] = [
        model_label(m, f) for m, f in zip(summary["model"], summary["features"])]
    return _pivot(summary, index=["target", "cutoff"], columns="variant",
                  n_folds=n_folds, latex=latex)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Beyond this many columns a table will not fit \textwidth, so it is promoted
# to a full-width float and shrunk. The ablation and results tables carry one
# column per (model, feature config) and cross this easily.
WIDE_COLUMNS = 6


def to_latex(table, caption, label, column_format=None, note=None, wide=None):
    """A booktabs table, in the style the manuscript uses.

    ``note`` is appended as a ``\\footnotesize`` line under the table --
    the place to say which population K was ranked over, or that a starred
    cell rests on fewer folds than the others.

    ``wide`` promotes the float to ``table*`` and shrinks the font. Left at
    ``None`` it is decided by the column count (:data:`WIDE_COLUMNS`), since
    these tables grow a column per model *and* feature config and a table
    that overflows the text block is not usable output.
    """
    if table.empty:
        return f"% {label}: no rows on disk yet\n"

    wide = len(table.columns) > WIDE_COLUMNS if wide is None else wide
    column_format = column_format or ("l" * table.index.nlevels
                                      + "c" * len(table.columns))
    # pandas warns that to_latex will move to the Styler implementation. The
    # arguments used here are the stable ones; silenced locally so the caller
    # still sees any other warning it raises.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        body = table.to_latex(escape=False, multirow=True,
                              column_format=column_format, caption=caption,
                              label=label)
    if note:
        body = body.replace(r"\end{table}",
                            r"\footnotesize " + note + "\n" + r"\end{table}")

    if wide:
        body = (body.replace(r"\begin{table}", r"\begin{table*}[t]")
                    .replace(r"\end{table}", r"\end{table*}")
                    .replace(r"\centering", "\\centering\n\\footnotesize", 1))
    return body


def write_table(table, stem, caption, label, results_dir="results", note=None,
                column_format=None, wide=None):
    """Write one table as both ``.tex`` (for the paper) and ``.csv`` (to read).

    Returns the paths written, or ``[]`` for an empty table -- a table with no
    rows is announced by the caller rather than written as an empty file that
    looks like a result.
    """
    if table.empty:
        return []

    tex_path = os.path.join(results_dir, f"table_{stem}.tex")
    csv_path = os.path.join(results_dir, f"table_{stem}.csv")
    with open(tex_path, "w") as handle:
        handle.write(to_latex(table, caption, label, column_format, note, wide))
    table.to_csv(csv_path)
    return [tex_path, csv_path]
