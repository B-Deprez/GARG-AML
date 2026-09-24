"""
Build the result tables from the tidy metrics.

Reads every ``results/<dataset>_<direction><suffix>_metrics.csv`` and writes
``results/table_<name>.tex`` plus a ``.csv`` twin of each. Nothing is
recomputed here -- this is the reporting step for numbers the model scripts
already produced, which is why it takes seconds and can be re-run after every
job finishes.

    python scripts/build_tables.py

What it writes
--------------
``table_results_<dataset>_<metric>``
    Every model on disk, GraphSAGE included, at the headline cut-offs and
    patterns, scaled by 100.
``table_ablation_<dataset>_<direction>_<metric>``
    The four feature configurations side by side, including the degree-only
    one, which carries no GARG-AML signal.
``table_alerts_<dataset>_<cutoff>_<target>_<metric>``
    The threshold-free alert-queue table: P@K / R@K / lift@K / TP@K over
    realistic queue sizes, from the pooled out-of-fold scores where they
    exist. Written beside its ``ties@K`` companion, always.
``table_variance_<dataset>_<metric>``
    Fold mean +/- std, with the fold count behind each mean.
``table_cost_<dataset>``
    GraphSAGE fit / inference seconds and peak host and GPU memory.
``table_sweep_<dataset>_<direction>_<metric>``
    Downstream performance against the Louvain resolution, with the
    no-Louvain arm first. Only written once at least two settings are on
    disk -- run the measure scripts on the ``_res<r>`` / ``_nolouvain``
    dataset names first.
``table_severance``
    Percentage of edges the pre-processing discards, per dataset and
    setting, from ``results/louvain_severance.csv``.
``table_splitting_<column>``
    What that discarding destroys: laundering attempts left undetectable,
    their two-hop path survival, and how many are split across communities
    -- by pattern type and resolution, from
    ``results/pattern_splitting_summary.csv`` (written by
    ``scripts/pattern_splitting.py``). It reads beside ``table_severance``:
    edges severed is the cost, attempts destroyed is what it buys.

Before the tables, it prints a coverage report of what is on disk, whose
``folds`` column says whether a row comes from a cross-validated run or from
a single split.

Gaps
----
A table with no rows on disk is announced and skipped, not written empty. A
cell the models could not be fitted on renders as ``--``; a mean resting on
fewer folds than the others is starred with its count. The gaps are reported
rather than hidden.
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd

from src.utils.reporting import (HEADLINE_CUTOFFS, HEADLINE_TARGETS,
                                 SPLITTING_COLUMNS,
                                 ablation_table, alert_table, coverage,
                                 cost_table, load_metrics, results_table,
                                 pattern_splitting_table, severance_table,
                                 sweep_table, ties_table, variance_table,
                                 write_table)
from src.utils.runtime import resolve_results_dir

# The datasets to build tables for. A missing one is skipped with a message.
DATASETS = ["HI-Small", "LI-Large"]

# Both threshold-free metrics are reported: AUC-PR is the primary one under
# 0.1 % prevalence, and AUC-ROC is included because the feature-group
# ablation is largely invisible on it, so quoting ROC alone would misread it.
METRICS = ["AUC_PR", "AUC_ROC"]

# The alert-queue tables are per (cut-off, pattern) cell; one table per cell
# in the whole grid would be 54 tables, so this is the headline slice. Each
# pattern appears at both 0.0 ("at least one laundering transaction") and
# 0.1, so the pair read side by side shows how sensitive the ranking is to
# where the propensity threshold sits, at the queue sizes an investigator
# actually works. Each cell is written as its own file.
ALERT_CELLS = [(0.0, "Is Laundering"),
               (0.0, "SCATTER-GATHER"),
               (0.0, "GATHER-SCATTER"),
               (0.1, "Is Laundering"),
               (0.1, "SCATTER-GATHER"),
               (0.1, "GATHER-SCATTER")]
ALERT_METRICS = ["P@K", "R@K", "lift@K", "TP@K"]

# Expected fold count, used only to mark a mean that rests on fewer. Keep in
# step with gargaml_tree.py's N_FOLDS; a wrong value here mislabels cells but
# changes no number.
N_FOLDS = 5

DIRECTIONS = ["undirected", "directed"]

RESULTS_DIR = resolve_results_dir()


def _slug(text):
    """Filename-safe, and \input-safe: LaTeX chokes on @ in a filename."""
    return (str(text).replace(" ", "-").replace(".", "")
            .replace("@", "at").replace("/", "-"))


def build_results_tables(df, dataset, written):
    for metric in METRICS:
        table = results_table(df, dataset, metric=metric, n_folds=N_FOLDS)
        if table.empty:
            print("  (no rows for "+metric+")")
            continue
        written += write_table(
            table, f"results_{dataset}_{metric}",
            caption=(f"{metric.replace('_', '-')} ($\\times$100) on {dataset}, "
                     "at the headline label cut-offs. Cells are the mean over "
                     f"{N_FOLDS} cross-validation folds $\\pm$ one standard "
                     "deviation; \\texttt{--} marks a cell with too few "
                     "positives to fit."),
            label=f"tab:results-{dataset.lower()}-{metric.lower()}",
            note=("A starred cell rests on fewer folds than the others; the "
                  "count is given. Feature configurations follow "
                  "\\texttt{src/utils/features.py}."),
            results_dir=RESULTS_DIR)
        print("  results ("+metric+"): "+str(table.shape))


def build_ablation_tables(df, dataset, written):
    for direction in DIRECTIONS:
        for metric in METRICS:
            table = ablation_table(df, dataset, direction, metric=metric,
                                   n_folds=N_FOLDS)
            if table.empty:
                continue
            written += write_table(
                table, f"ablation_{dataset}_{direction}_{metric}",
                caption=(f"Feature-group ablation, {direction} {dataset}, "
                         f"{metric.replace('_', '-')}. \\emph{{Degree-only}} "
                         "contains no GARG-AML signal; \\emph{blocks only} "
                         "is the raw block densities and sizes before "
                         "aggregation; the unlabelled model is the published "
                         "feature set."),
                label=f"tab:ablation-{dataset.lower()}-{direction}-{metric.lower()}",
                note=("The degree-only configuration is computed on the "
                      "Louvain-reduced graph, so it is not confounded with the "
                      "edge-removal step. It is direction-free and therefore "
                      "identical in both tables."),
                results_dir=RESULTS_DIR)
            print("  ablation ("+direction+", "+metric+"): "+str(table.shape))


def build_alert_tables(df, dataset, written):
    for cutoff, target in ALERT_CELLS:
        for metric in ALERT_METRICS:
            table = alert_table(df, dataset, cutoff, target, metric=metric,
                                n_folds=N_FOLDS)
            if table.empty:
                continue
            # The population is per row, not per table: a model with a
            # pooled out-of-fold pass is ranked over every account, one on a
            # single split over its test slice only. The table prints it per
            # row, so the caption points there rather than asserting one
            # population for all of them.
            populations = table["ranked over"].nunique()
            population = ("the population given in the first column; rows "
                          "differ because models evaluated under "
                          "cross-validation are ranked over pooled "
                          "out-of-fold scores"
                          if populations > 1 else
                          "the population given in the first column")
            written += write_table(
                table, f"alerts_{dataset}_{_slug(cutoff)}_{_slug(target)}_{_slug(metric)}",
                caption=(f"{metric} at investigator alert-queue sizes, "
                         f"{dataset}, {target} at cut-off {cutoff}. Ranked over "
                         f"{population}."),
                label=f"tab:alerts-{dataset.lower()}-{_slug(cutoff)}-{_slug(target)}-{_slug(metric)}",
                note=("Read beside the tie diagnostic: where \\texttt{ties@K} "
                      "exceeds 1 the top-$K$ set is not determined by the "
                      "scores alone."),
                results_dir=RESULTS_DIR)
            print("  alerts ("+target+" @"+str(cutoff)+", "+metric+"): "+str(table.shape))

        ties = ties_table(df, dataset, cutoff, target, n_folds=N_FOLDS)
        if not ties.empty:
            written += write_table(
                ties, f"alerts_{dataset}_{_slug(cutoff)}_{_slug(target)}_ties",
                caption=(f"Tie diagnostic for {dataset}, {target} at cut-off "
                         f"{cutoff}: the size of the score group straddling the "
                         "top-$K$ cut. A value of 1 means the queue is "
                         "determined by the scores."),
                label=f"tab:ties-{dataset.lower()}-{_slug(cutoff)}-{_slug(target)}",
                results_dir=RESULTS_DIR)
            print("  ties ("+target+" @"+str(cutoff)+"): "+str(ties.shape))


def build_sweep_tables(df, dataset, written):
    """The Louvain resolution sweep, one table per (direction, metric).

    Silently absent until at least two Louvain settings have been run --
    sweep_table returns empty rather than presenting a single arm as a
    sensitivity analysis.
    """
    for direction in DIRECTIONS:
        for metric in METRICS:
            table = sweep_table(df, dataset, direction, metric=metric,
                                n_folds=N_FOLDS)
            if table.empty:
                continue
            written += write_table(
                table, f"sweep_{dataset}_{direction}_{metric}",
                caption=(f"Sensitivity of {metric.replace('_', '-')} to the "
                         f"Louvain pre-processing, {direction} {dataset}. "
                         "Columns run from no reduction at all to the most "
                         "aggressive setting; \\emph{r=10} is the published "
                         "choice. The published feature configuration is used "
                         "throughout, so this isolates the pre-processing from "
                         "the feature-group sensitivity reported separately."),
                label=f"tab:sweep-{dataset.lower()}-{direction}-{metric.lower()}",
                note=("Cells are deliberately not bolded: the question is "
                      "whether the choice of resolution moves the result, not "
                      "which resolution to select on the evaluation data."),
                # Three index levels of long model labels plus a column per
                # setting overflows \textwidth well before the column count
                # alone would trigger the automatic promotion.
                wide=True, results_dir=RESULTS_DIR)
            print("  sweep ("+direction+", "+metric+"): "+str(table.shape))


def build_dataset(df, dataset, written):
    if df[df["dataset"] == dataset].empty:
        print("\n### "+dataset+" -- SKIPPED: no tidy metrics on disk ###")
        return

    print("\n=== "+dataset+" ===")
    build_results_tables(df, dataset, written)
    build_ablation_tables(df, dataset, written)
    build_alert_tables(df, dataset, written)
    build_sweep_tables(df, dataset, written)

    for metric in METRICS:
        table = variance_table(df, dataset, metric=metric, n_folds=N_FOLDS)
        if table.empty:
            continue
        written += write_table(
            table, f"variance_{dataset}_{metric}",
            caption=(f"Fold-to-fold spread of {metric.replace('_', '-')} on "
                     f"{dataset}: mean $\\pm$ standard deviation over "
                     f"{N_FOLDS} stratified folds."),
            label=f"tab:variance-{dataset.lower()}-{metric.lower()}",
            note=("The evaluation is transductive: neighbourhood summary "
                  "features are computed on the full graph before folding."),
            results_dir=RESULTS_DIR)
        print("  variance ("+metric+"): "+str(table.shape))

    costs = cost_table(df, dataset, n_folds=N_FOLDS)
    if not costs.empty:
        written += write_table(
            costs, f"cost_{dataset}",
            caption=(f"Training and inference cost on {dataset}. GARG-AML has "
                     "no fit stage, which is why fit time is reported apart "
                     "from preprocessing and inference."),
            label=f"tab:cost-{dataset.lower()}",
            results_dir=RESULTS_DIR)
        print("  cost: "+str(costs.shape))


def main():
    df = load_metrics(results_dir=RESULTS_DIR)
    if df.empty:
        print("No tidy metrics files in "+RESULTS_DIR+"/. Run a model script "
              "first (e.g. scripts/gargaml_tree.py).")
        return

    print("=== Coverage: what is on disk ===")
    report = coverage(df)
    print(report.to_string(index=False))

    stale = report[report["folds"] == "single"]
    if len(stale):
        print("\n"+str(len(stale))+" of "+str(len(report))+" model/config "
              "combinations have no fold column: they come from a single "
              "train/test split. Tables built from them describe that split "
              "rather than cross-validation; re-run gargaml_tree.py with "
              "N_FOLDS >= 2 for fold-based numbers.")

    report.to_csv(RESULTS_DIR+"/table_coverage.csv", index=False)

    written = []

    # Edge severance is per (dataset, setting) and comes from the measure
    # scripts' own log, not from the metrics, so it is built once rather
    # than inside the per-dataset loop.
    severance = severance_table(results_dir=RESULTS_DIR)
    if severance.empty:
        print("\nNo "+RESULTS_DIR+"/louvain_severance.csv yet -- run a measure "
              "script to record how much the Louvain step discards.")
    else:
        written += write_table(
            severance, "severance",
            caption=("Percentage of edges discarded by the Louvain "
                     "pre-processing, per dataset and resolution. "
                     "\\emph{r=10} is the published choice."),
            label="tab:severance",
            note=("Every inter-community edge is dropped before scoring, so "
                  "this is the fraction of the graph the second-order "
                  "neighbourhoods never see."),
            results_dir=RESULTS_DIR)
        print("\nseverance: "+str(severance.shape))

    # What that discarding destroys. Also outside the per-dataset loop: the
    # summary already spans datasets, and the table's axes are pattern type
    # and resolution rather than anything from the metrics frame.
    for column, (phrase, _) in SPLITTING_COLUMNS.items():
        table = pattern_splitting_table(results_dir=RESULTS_DIR, column=column)
        if table.empty:
            continue
        written += write_table(
            table, f"splitting_{column}",
            caption=("Percentage " + phrase + " after the Louvain "
                     "pre-processing, by pattern type and resolution. "
                     "GARG-AML targets GATHER-SCATTER and SCATTER-GATHER, "
                     "which lead the table."),
            label=f"tab:splitting-{column.replace('_', '-')}",
            note=("\\texttt{--} marks a pattern type with no two-hop "
                  "structure to lose (FAN-OUT, FAN-IN), not a zero. "
                  "\\emph{no Louvain} is the control: nothing is "
                  "partitioned, so every attempt survives by construction."),
            results_dir=RESULTS_DIR)
        print("splitting ("+column+"): "+str(table.shape))
    if not written:
        print("\nNo "+RESULTS_DIR+"/pattern_splitting_summary.csv yet -- run "
              "scripts/pattern_splitting.py.")

    for dataset in DATASETS:
        build_dataset(df, dataset, written)

    print("\nwrote "+str(len(written))+" files to "+RESULTS_DIR+"/ "
          "("+str(len(written) // 2)+" tables, .tex + .csv each)")


if __name__ == "__main__":
    main()
