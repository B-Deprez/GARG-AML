"""
Build the revision's tables from the tidy metrics (P1, P2, P3, P7).

Reads every ``results/<dataset>_<direction><suffix>_metrics.csv`` and writes
``results/table_<name>.tex`` plus a ``.csv`` twin of each. Nothing is
recomputed here -- this is the reporting step for numbers the model scripts
already produced, which is why it takes seconds and can be re-run after every
job finishes.

    python scripts/build_tables.py

What it writes, and which reviewer point each answers
-----------------------------------------------------
``table_results_<dataset>_<metric>``   (P1, R2-M1)
    Tables 10-11 across every model on disk, GraphSAGE included, at the
    headline cut-offs and patterns, scaled by 100.
``table_ablation_<dataset>_<direction>_<metric>``   (P3, R2-M5)
    The four feature configs side by side. ``Degree-only`` carries no
    GARG-AML signal, so the gap between it and the published model is the
    answer to "significantly reducing false positives".
``table_alerts_<dataset>_<cutoff>_<target>_<metric>``   (P2, R1-5, R2-M4)
    The threshold-free alert-queue table: P@K / R@K / lift@K / TP@K over
    realistic queue sizes, from the pooled out-of-fold scores where they
    exist. Written beside its ``ties@K`` companion, always.
``table_variance_<dataset>_<metric>``   (P7, R2-M6)
    Fold mean +/- std, with the fold count behind each mean.
``table_cost_<dataset>``   (P1, scalability)
    GraphSAGE fit / inference seconds and peak host and GPU memory.

Before the tables, it prints a coverage report. Read it first: while the
re-runs are outstanding most cells come from the 15 Sep single-split grid,
and a table built from those is a snapshot of the old run, not of the code.
The ``folds`` column says ``single`` for exactly those rows.

Gaps
----
A table with no rows on disk is announced and skipped, not written empty. A
cell the models could not be fitted on renders as ``--``; a mean resting on
fewer folds than the others is starred with its count. None of that is
cosmetic -- the revision requires the gaps to be reported.
"""

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd

from src.utils.reporting import (HEADLINE_CUTOFFS, HEADLINE_TARGETS,
                                 ablation_table, alert_table, coverage,
                                 cost_table, load_metrics, results_table,
                                 ties_table, variance_table, write_table)

# The datasets to build tables for. A missing one is skipped with a message,
# so leaving LI-Large here before its run finishes costs nothing.
DATASETS = ["HI-Small", "LI-Large"]

# Reported for both threshold-free metrics: AUC-PR is the primary one under
# 0.1 % prevalence, and AUC-ROC is included because the task-3 ablation is
# *invisible* on it -- degree-only is competitive and sometimes better there.
# Quoting only ROC would let a reader conclude the opposite of the truth, so
# the pair is the point, not a redundancy.
METRICS = ["AUC_PR", "AUC_ROC"]

# The alert-queue tables are per (cut-off, pattern) cell; one table per cell
# in the whole grid would be 45 tables, so this is the headline slice.
ALERT_CELLS = [(0.1, "Is Laundering"),
               (0.1, "SCATTER-GATHER"),
               (0.1, "GATHER-SCATTER")]
ALERT_METRICS = ["P@K", "R@K", "lift@K", "TP@K"]

# Expected fold count, used only to mark a mean that rests on fewer. Keep in
# step with gargaml_tree.py's N_FOLDS; a wrong value here mislabels cells but
# changes no number.
N_FOLDS = 5

DIRECTIONS = ["undirected", "directed"]


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
                  "\\texttt{src/utils/features.py}."))
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
                         "contains no GARG-AML signal; \\emph{{blocks only}} "
                         "is the raw block densities and sizes before "
                         "aggregation; the unlabelled model is the published "
                         "feature set."),
                label=f"tab:ablation-{dataset.lower()}-{direction}-{metric.lower()}",
                note=("The degree-only configuration is computed on the "
                      "Louvain-reduced graph, so it is not confounded with the "
                      "edge-removal step. It is direction-free and therefore "
                      "identical in both tables."))
            print("  ablation ("+direction+", "+metric+"): "+str(table.shape))


def build_alert_tables(df, dataset, written):
    for cutoff, target in ALERT_CELLS:
        for metric in ALERT_METRICS:
            table = alert_table(df, dataset, cutoff, target, metric=metric,
                                n_folds=N_FOLDS)
            if table.empty:
                continue
            # The population is per row, not per table: a model with a
            # pooled out-of-fold pass is ranked over every account, one
            # still on the single split over its test slice only. The
            # table prints it per row, so the caption points there rather
            # than asserting one population for all of them.
            populations = table["ranked over"].nunique()
            population = ("the population given in the first column; rows "
                          "differ because not every model has been re-run "
                          "under cross-validation yet"
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
                      "scores alone."))
            print("  alerts ("+target+" @"+str(cutoff)+", "+metric+"): "+str(table.shape))

        ties = ties_table(df, dataset, cutoff, target, n_folds=N_FOLDS)
        if not ties.empty:
            written += write_table(
                ties, f"alerts_{dataset}_{_slug(cutoff)}_{_slug(target)}_ties",
                caption=(f"Tie diagnostic for {dataset}, {target} at cut-off "
                         f"{cutoff}: the size of the score group straddling the "
                         "top-$K$ cut. A value of 1 means the queue is "
                         "determined by the scores."),
                label=f"tab:ties-{dataset.lower()}-{_slug(cutoff)}-{_slug(target)}")
            print("  ties ("+target+" @"+str(cutoff)+"): "+str(ties.shape))


def build_dataset(df, dataset, written):
    if df[df["dataset"] == dataset].empty:
        print("\n### "+dataset+" -- SKIPPED: no tidy metrics on disk ###")
        return

    print("\n=== "+dataset+" ===")
    build_results_tables(df, dataset, written)
    build_ablation_tables(df, dataset, written)
    build_alert_tables(df, dataset, written)

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
                  "features are computed on the full graph before folding."))
        print("  variance ("+metric+"): "+str(table.shape))

    costs = cost_table(df, dataset, n_folds=N_FOLDS)
    if not costs.empty:
        written += write_table(
            costs, f"cost_{dataset}",
            caption=(f"Training and inference cost on {dataset}. GARG-AML has "
                     "no fit stage, which is why fit time is reported apart "
                     "from preprocessing and inference."),
            label=f"tab:cost-{dataset.lower()}")
        print("  cost: "+str(costs.shape))


def main():
    df = load_metrics()
    if df.empty:
        print("No tidy metrics files in results/. Run a model script first "
              "(e.g. scripts/gargaml_tree.py).")
        return

    print("=== Coverage: what is on disk ===")
    report = coverage(df)
    print(report.to_string(index=False))

    stale = report[report["folds"] == "single"]
    if len(stale):
        print("\n"+str(len(stale))+" of "+str(len(report))+" model/config "
              "combinations have no fold column, i.e. they predate task 7's "
              "cross-validation. Tables built from them describe that earlier "
              "single-split run; re-run gargaml_tree.py with N_FOLDS >= 2 "
              "before quoting them.")

    report.to_csv("results/table_coverage.csv", index=False)

    written = []
    for dataset in DATASETS:
        build_dataset(df, dataset, written)

    print("\nwrote "+str(len(written))+" files to results/ "
          "("+str(len(written) // 2)+" tables, .tex + .csv each)")


if __name__ == "__main__":
    main()
