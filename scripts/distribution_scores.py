import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.methods.gargaml_scores import define_gargaml_scores
from scripts.gargaml_tree_synthetic import holdout_test_index, merge_labels
from src.utils.evaluation import (CUT_OFFS, LEGACY_METRICS, SEED,
                                  evaluate_scores, fold_assignments, folds_path,
                                  metric_names, metric_records, nan_metrics,
                                  read_folds, write_metrics)
from src.utils.graph_processing import strip_resolution
from src.utils.naming import gargaml_key
from src.utils.runtime import resolve_results_dir, select_datasets, write_csv
from functools import lru_cache
import pandas as pd
import timeit

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = resolve_results_dir()

# Per-fold breakdown of the base GARG-AML score. The score is deterministic
# and nothing is fitted, so the pooled out-of-fold value equals the
# full-population value by construction, and the per-fold spread is pure
# evaluation-slice noise -- the noise floor the tree/boost fold spread is
# read against. The partition is read from results/<dataset>_folds.csv
# (written by gargaml_tree.py, N_FOLDS >= 2), never re-derived, so the base
# score is paired fold for fold with the tree models and GraphSAGE.
#
# Set to False to skip the per-fold rows. Also falls back on its own when no
# partition exists (the synthetic grid, or an IBM run with N_FOLDS = 0);
# full-population numbers are still written either way.
USE_FOLDS = True

# The Louvain sensitivity sweep (task 4) compares the pure score alone, directed
# against undirected, across the resolution setting; no model is fitted on its
# arms. ``GARGAML_DATASET=louvain_sweep`` expands to these names, the way
# ``synthetic`` stands for the synthetic grid. The published setting (plain
# "HI-Small", resolution 10) is the sweep's own point and is not rerun here.
# Each arm has no fold partition (gargaml_tree.py is not run on it), so it gets
# full-population metrics only, which for an unfitted score are the pooled
# out-of-fold numbers anyway.
LOUVAIN_SWEEP = ["HI-Small_res1", "HI-Small_res5", "HI-Small_res20",
                 "HI-Small_res50", "HI-Small_nolouvain"]

# This file's convention for "not available" is -1, not NaN: results are
# logged as repr()'d Python literals and read back with a bare eval() in
# notebooks/VisualisationResults.ipynb, which has no `nan` name in scope.
# _sanitise substitutes -1 before anything gets str()'d into a log line.
def _sanitise(value):
    return -1 if isinstance(value, float) and np.isnan(value) else value

# A synthetic result cell uses the layout and key spelling of the cells
# scripts/gargaml_tree_synthetic*.py write (Precision/F1 capitalised, every
# ranking metric beside them), so the notebook reads the base score and the
# tree models the same way. Plain floats, so a NumPy 2 repr does not put
# np.float64(...) into the file.
SYNTHETIC_KEY_RENAME = {"precision": "Precision", "f1": "F1"}

def _synthetic_cell(metrics):
    return {SYNTHETIC_KEY_RENAME.get(k, k): _sanitise(float(v)) for k, v in metrics.items()}

def divergence_metric(dist_0, dist_1):
    mean_0 = np.mean(dist_0)
    variance_0 = np.var(dist_0)
    mean_1 = np.mean(dist_1)
    variance_1 = np.var(dist_1)
    return (mean_0 - mean_1)**2 + 0.5*(variance_0 + variance_1)

def lift_curve_values(y_val, y_pred, steps):
    vals_lift = []

    df_lift = pd.DataFrame()
    df_lift['Real'] = y_val
    df_lift['Pred'] = y_pred
    df_lift.sort_values('Pred',
                        ascending=False,
                        inplace=True)

    global_ratio = df_lift['Real'].sum() / len(df_lift['Real'])

    for step in steps:
        data_len = int(np.ceil(step*len(df_lift)))
        data_value = df_lift.iloc[data_len-1]['Pred']
        data_lift = df_lift[df_lift['Pred'] >= data_value]
        val_lift = data_lift['Real'].sum()/len(data_lift)
        vals_lift.append(val_lift/global_ratio)

    return(vals_lift)

def distribution_scores_IBM_plots(dataset, results_df, str_directed, str_supervised):
    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)

    from_data = transactions_df_extended[["Account", "From Bank"]].drop_duplicates()
    from_data.columns = ["Account", "Bank"]
    to_data = transactions_df_extended[["Account.1", "To Bank"]].drop_duplicates()
    to_data.columns = ["Account", "Bank"]

    print("="*10)
    print("Data loaded")

    cut_offs = CUT_OFFS # the canonical sweep -- see src/utils/evaluation.py
    columns = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

    n = len(cut_offs)
    m = len(columns)

    divergence_matrix = np.zeros((n, m))

    fig, axes = plt.subplots(n, m, figsize = (n*5, m))

    for i in range(n):
        cut_off = cut_offs[i]
        for j in range(m):
            column = columns[j]
            print(cut_off, column)
            laundering_combined["Label"] = ((laundering_combined[column]>cut_off)*1).values

            labels = []
            for node in results_df.index:
                label = int(laundering_combined.loc[node]["Label"])
                labels.append(label)

            results_df["Label"] = labels
        
            label_0 = results_df[results_df["Label"] == 0]["GARGAML"]
            label_1 = results_df[results_df["Label"] == 1]["GARGAML"]

            all_data = np.concatenate([label_0, label_1])
            bins = np.histogram_bin_edges(all_data, bins=20)

            axes[i, j].hist(label_0, bins=bins, alpha=0.5, label='Label 0', density=True)
            axes[i, j].hist(label_1, bins=bins, alpha=0.5, label='Label 1', density=True)

            divergence = divergence_metric(label_0, label_1)
            divergence_matrix[i, j] = divergence

    divergence_df = pd.DataFrame(divergence_matrix, columns=columns, index=cut_offs)
    divergence_df.to_csv(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_supervised+"_"+str_directed+"_combined_divergence.csv")

    print("="*10)
    print("Divergence saved")

    for axis, col in zip(axes[0], columns):
        axis.set_title(col)

    for axis, row in zip(axes[:,0], cut_offs):
        axis.set_ylabel(row, size='large')

    if supervised:
        fig.suptitle('Distribution of '+ str_directed +' GARGAML Scores by Label for data set: '+ dataset)
    else:
        fig.suptitle('Distribution of '+ str_directed +' anomaly scores by Label for data set: '+ dataset)
    fig.tight_layout()
    plt.savefig(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_supervised+"_"+str_directed+"_combined_histogram.pdf")
    plt.close()

    print("="*10)
    print("Figure distributions saved")

    fig, axes = plt.subplots(n, m, figsize = (n*5, m))
    values = np.linspace(0.01, 1, 100)

    for i in range(n):
        cut_off = cut_offs[i]
        for j in range(m):
            column = columns[j]
            print(cut_off, column)
            laundering_combined["Label"] = ((laundering_combined[column]>cut_off)*1).values

            labels = []
            for node in results_df.index:
                label = int(laundering_combined.loc[node]["Label"])
                labels.append(label)

            results_df["Label"] = labels
        
            lift = lift_curve_values(results_df["Label"], results_df["GARGAML"], values)

            axes[i, j].plot(values, lift)

    for axis, col in zip(axes[0], columns):
        axis.set_title(col)

    for axis, row in zip(axes[:,0], cut_offs):
        axis.set_ylabel(row, size='large')

    if supervised:
        fig.suptitle('Lift curve of '+ str_directed +' GARGAML Scores by Label for data set: '+ dataset)
    else:
        fig.suptitle('Lift curve of '+ str_directed +' anomaly scores by Label for data set: '+ dataset)

    fig.tight_layout()
    plt.savefig(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_supervised+"_"+str_directed+"_combined_lift.pdf")
    plt.close()

    print("="*10)
    print("Figure lift saved")

def _fold_records(labels_gargaml_full, y_true, folds_df, cut_off, column, context):
    """Per-fold and pooled out-of-fold rows for one (cut-off, target) cell.

    Returns ``[]`` when there is no partition for this cell -- either no
    folds file at all (the synthetic grid, or an N_FOLDS=0 IBM run) or a
    cell with too few positives to fold. The caller's full-population row
    is written either way, so an empty return here costs nothing.

    The pooled row (``fold = -1``) is computed over the union of the test
    folds rather than over the whole frame. Those two populations should
    coincide, but need not: this script's ``labels_gargaml_full`` is an
    outer merge, so it can hold accounts that never reached
    gargaml_tree.py's feature table and therefore never entered the
    partition. ``n_test`` is written on every row so such a mismatch is
    visible rather than assumed away.
    """
    assignments = fold_assignments(folds_df, cut_off, column)
    if not assignments:
        return []

    y_series = pd.Series(y_true, index=labels_gargaml_full.index)
    score_series = labels_gargaml_full["GARGAML"]

    records = []
    pooled = []
    for fold in sorted(assignments):
        accounts = labels_gargaml_full.index.intersection(assignments[fold])
        pooled.append(accounts)
        y_f, s_f = y_series.loc[accounts], score_series.loc[accounts]
        try:
            metrics = evaluate_scores(y_f.values, s_f.values)
            status = "ok"
        except Exception as exc:  # a fold with no positives cannot be scored
            print("    fold "+str(fold)+" skipped: "+repr(exc))
            metrics, status = nan_metrics(), "skipped: "+str(exc)
        records += metric_records(metrics, status=status, n_test=len(accounts),
                                  n_pos=int(y_f.sum()), fold=fold, **context)

    accounts = labels_gargaml_full.index[
        labels_gargaml_full.index.isin(np.concatenate([a.values for a in pooled]))]
    y_p, s_p = y_series.loc[accounts], score_series.loc[accounts]
    try:
        metrics = evaluate_scores(y_p.values, s_p.values)
        status = "ok"
    except Exception as exc:
        metrics, status = nan_metrics(), "skipped: "+str(exc)
    records += metric_records(metrics, status=status, n_test=len(accounts),
                              n_pos=int(y_p.sum()), fold=-1, **context)
    return records


@lru_cache(maxsize=1)
def ibm_labels(base):
    """Account labels of an IBM dataset, read from its transactions file.

    Every pre-processing arm of a dataset ("HI-Small_res20", ...) shares the
    labels of its base, and the sweep scores each arm in both directions, so
    the parse of the transactions file is done once per base rather than once
    per (arm, direction).
    """
    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans = "data/"+base+"_Trans.csv",
        path_patterns = "data/"+base+"_Patterns.txt"
    )
    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)
    return laundering_combined

def distribution_scores_IBM(dataset, results_df, str_directed, str_supervised):
    laundering_combined = ibm_labels(strip_resolution(dataset))

    labels_gargaml_full = laundering_combined.merge(results_df[["GARGAML"]], left_index=True, right_index=True, how="outer").fillna(-1)

    # Evaluated on every account, not on a held-out slice: this score is never
    # fit to anything (same as gargaml_IF.py, for the same reason).
    y_pred = labels_gargaml_full["GARGAML"].values

    # The fold partition, if one was written. None is a normal state, not an
    # error -- see USE_FOLDS.
    folds_df = read_folds(dataset, results_dir=RESULTS_DIR) if USE_FOLDS else None
    if folds_df is None:
        print("No fold partition at "+folds_path(dataset, results_dir=RESULTS_DIR)+
              ": reporting full-population metrics only (the per-fold "
              "breakdown needs gargaml_tree.py run with N_FOLDS >= 2 first).")
    else:
        print("Per-fold breakdown on the persisted fold partition, "
              +str(folds_df["fold"].nunique())+" folds. The base score is not "
              "fitted, so the pooled out-of-fold row equals the full-population "
              "row and the fold spread is the evaluation-slice noise floor.")

    model_key = gargaml_key("base", str_directed == "directed")
    records = []

    cut_offs = CUT_OFFS # the canonical sweep -- see src/utils/evaluation.py
    columns = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

    n = len(cut_offs)
    m = len(columns)

    for i in range(n):
        cut_off = cut_offs[i]
        for j in range(m):
            column = columns[j]
            print(cut_off, column)
            y_true = ((labels_gargaml_full[column]>cut_off)*1).values

            context = dict(dataset=dataset, direction=str_directed,
                           model=model_key, features="score", cutoff=cut_off,
                           target=column, seed=SEED)

            try:
                # No natural 0/1 prediction for a raw score -- precision/f1
                # come back NaN rather than invented at some threshold.
                metrics = evaluate_scores(y_true, y_pred)
                result_list = [_sanitise(metrics[name]) for name in metric_names()]
                status = "ok"
            except Exception as exc:
                print("    skipped: "+repr(exc))
                result_list = [-1 for _ in metric_names()]
                metrics, status = nan_metrics(), "skipped: "+str(exc)

            # The full-population row keeps fold = NaN: it is neither one of
            # the folds nor the pooled pass, and it is the number written to
            # the .txt log below.
            records += metric_records(metrics, status=status,
                                      n_test=len(y_true), n_pos=int(y_true.sum()),
                                      fold=np.nan, **context)
            records += _fold_records(labels_gargaml_full, y_true, folds_df,
                                     cut_off, column, context)

            # Fragile log format, read by VisualisationResults.ipynb: that
            # notebook finds the list with line.split(': ', maxsplit=3), so
            # a 4th ': ' anywhere on the line (e.g. a nested dict) truncates
            # it before the real data -- a flat list of numbers has none.
            # It then parses the whole line with line.split('_'), so an
            # extra '_' in the label text shifts every index after it.
            # A sweep arm's name carries an extra '_' that would shift the
            # parse above, and its numbers live in the tidy file.
            if strip_resolution(dataset) == dataset:
                with open(RESULTS_DIR+'/results_performance_IBM_'+str_directed+'.txt', 'a') as f:
                    f.write(dataset+'_'+column+'_'+str(cut_off)+' [precision, F1, AUC-ROC, AUC-PR, then '
                            +'ranking metrics in a fixed order, see evaluation.py]: '
                            +str(result_list)+'\n')

    # Tidy frame only. The base score has no
    # <dataset>_<metric>_<model>_<direction>_combined.csv matrices behind it
    # -- its numbers are the .txt log above -- so emitting matrices would
    # create files nothing reads.
    write_metrics(records, dataset, str_directed, suffix="_base",
                  results_dir=RESULTS_DIR, write_matrices=False)

def plot_distribution_synthetic(laundering_combined, columns, str_directed, str_supervised):
    n = len(columns)

    fig, axes = plt.subplots(n//2, n//2+n%2, figsize = (3*n, 1.5*n))

    for i in range(n):
        column = columns[i]
        laundering_combined["Label"] = laundering_combined[column].values

        label_0 = laundering_combined[laundering_combined["Label"] == 0]["GARGAML"]
        label_1 = laundering_combined[laundering_combined["Label"] == 1]["GARGAML"]

        all_data = np.concatenate([label_0, label_1])
        bins = np.histogram_bin_edges(all_data, bins=20)

        axes[i//2, i%2].hist(label_0, bins=bins, alpha=0.5, label='Other', density=True)
        axes[i//2, i%2].hist(label_1, bins=bins, alpha=0.5, label=column, density=True)

        axes[i//2, i%2].legend()
        axes[i//2, i%2].set_xlabel('GARG-AML score')
        axes[i//2, i%2].set_ylabel('Relative Frequency')
        axes[i//2, i%2].set_title(column)
    fig.tight_layout()
    plt.savefig(RESULTS_DIR+"/synthetic_GARGAML_"+str_supervised+"_"+str_directed+"_histogram.pdf")
    plt.close()

def plot_lift_synthetic(laundering_combined, columns, str_directed, str_supervised):
    n = len(columns)

    fig, axes = plt.subplots(n//2, n//2+n%2, figsize = (3*n, 1.5*n))
    values = np.linspace(0.01, 1, 100)

    for i in range(n):
        column = columns[i]
        laundering_combined["Label"] = laundering_combined[column].values
        
        lift = lift_curve_values(laundering_combined["Label"], laundering_combined["GARGAML"], values)

        axes[i//2, i%2].plot(values, lift)
        axes[i//2, i%2].set_xlabel('Percentage of data')
        axes[i//2, i%2].set_ylabel('Lift')
        axes[i//2, i%2].set_title(column)

    fig.tight_layout()
    plt.savefig(RESULTS_DIR+"/synthetic_GARGAML_"+str_supervised+"_"+str_directed+"_lift.pdf")
    plt.close()


# The rule behind the synthetic base score's Precision and F1: flag a node when
# its score is above this. A raw score has no 0/1 prediction of its own, so
# this is a stated cut, not something the model predicts -- the table labels it
# as such. 0.5 is the cut the published synthetic table used (it reproduces
# every published Precision/F1 of the two score rows to three decimals).
# Threshold-free metrics (AUC, P@K) do not depend on it.
SCORE_THRESHOLD = 0.5

def _score_metrics(frame, column):
    """Base-score metrics for one label over the rows of ``frame``.

    Precision and F1 use the ``GARGAML > SCORE_THRESHOLD`` rule; the IBM path
    (:func:`distribution_scores_IBM`) still reports them as NaN.
    """
    try:
        score = frame["GARGAML"].values
        return evaluate_scores((frame[column]*1).values, score,
                               y_pred=(score > SCORE_THRESHOLD).astype(int))
    except Exception as exc:
        print("    skipped: "+repr(exc))
        return nan_metrics()

def distribution_scores_synthetic(dataset, results_df, str_directed, str_supervised):
    """Base-score metrics on one synthetic dataset, un-folded.

    Cross-validation is scoped to the IBM data: the 66 synthetic datasets
    keep the single 70/30 split, and their variance comes from the
    66-dataset spread that feeds the Friedman/Nemenyi analysis.

    ``"score"`` is computed on the tree models' 30% test rows for each label
    (gargaml_tree_synthetic.py's :func:`holdout_test_index`), so the base
    score and the trees are ranked over the same nodes. That matters for the
    fixed-K metrics: an injected pattern labels ~31 nodes (median), the test
    rows hold ~9 of them, so P@10 and R@10 over every node are a different
    quantity -- R@10 is capped near 0.32 there and not on the test rows.
    ``"score_full"`` keeps the every-node figure the published tables used.

    Returns ``{pattern: {"score": cell, "score_full": cell}}``, the tree
    scripts' cell layout.
    """
    columns = ['laundering', 'separate', 'new_mules', 'existing_mules']
    # Nodes without connections are not smurfing (fillna(-1) in merge_labels).
    laundering_combined = merge_labels(results_df, dataset)

    plot_distribution_synthetic(laundering_combined, columns, str_directed, str_supervised)

    plot_lift_synthetic(laundering_combined, columns, str_directed, str_supervised)


    results = dict()
    for column in columns:
        print(column)
        test_rows = holdout_test_index(laundering_combined, column)
        metrics = _score_metrics(laundering_combined.loc[test_rows], column)
        metrics_full = _score_metrics(laundering_combined, column)

        results[column] = {"score": _synthetic_cell(metrics),
                           "score_full": _synthetic_cell(metrics_full)}
        for name in LEGACY_METRICS + ["P@10"]:
            print(name+": ", metrics[name], "(every node: "+str(metrics_full[name])+")")

    return results

def general_calculation(dataset, directed, supervised, score_type):
    str_directed = "directed" if directed else "undirected"
    str_supervised = "supervised" if supervised else "unsupervised"

    if supervised:
        # The synthetic undirected measures carry a "_parallel" suffix
        # (gargaml_undirected_synth.py); the IBM ones and every directed file
        # do not. Same rule as gargaml_tree_synthetic*.py's data_preparation.
        suffix = "_parallel" if dataset.startswith("synthetic") and not directed else ""
        results_df_measures = pd.read_csv(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_directed+suffix+".csv")
        results_df = define_gargaml_scores(results_df_measures, directed=directed, score_type=score_type)

    else:
        results_df = pd.read_csv(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_directed+"_IF.csv")
        results_df = results_df.set_index("node")
        results_df = results_df[["anomaly_score"]]
        results_df["anomaly_score"] = results_df["anomaly_score"]*(-1)
        results_df.columns = ["GARGAML"]

    if strip_resolution(dataset) in ["HI-Small", "LI-Large"]:
        distribution_scores_IBM(dataset, results_df, str_directed, str_supervised)

    elif dataset[:min(9, len(dataset))] == "synthetic": #use min in case the string is shorter than 9
        return distribution_scores_synthetic(dataset, results_df, str_directed, str_supervised)

    else:
        raise ValueError("Invalid dataset")

def benchmark_synthetic(
        directed, 
        supervised,
        score_type
):
    str_directed = "directed" if directed else "undirected"
    str_supervised = "supervised" if supervised else "unsupervised"

    n_nodes_list = [100, 10000, 100000]
    m_edges_list = [1, 2, 5] # BA/WS: edges attached per new node
    p_edges_list = [0.001, 0.01] # ER/WS: edge probability
    generation_method_list = [
        'Barabasi-Albert', 
        'Erdos-Renyi', 
        'Watts-Strogatz'
        ]
    n_patterns_list = [3, 5]

    results_dict = {}
    for n_nodes in n_nodes_list:
        for n_patterns in n_patterns_list:
            if n_patterns <= 0.06*n_nodes:
                for generation_method in generation_method_list:
                    if generation_method == 'Barabasi-Albert':
                        p_edges = 0
                        for m_edges in m_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            print("====", string_name, "====")
                            results_dict[string_name] = general_calculation(string_name, directed, supervised, score_type)
                    if generation_method == 'Erdos-Renyi':
                        m_edges = 0
                        for p_edges in p_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            print("====", string_name, "====")
                            results_dict[string_name] = general_calculation(string_name, directed, supervised, score_type)

                    if generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                                print("====", string_name, "====")
                                results_dict[string_name] = general_calculation(string_name, directed, supervised, score_type)

    # One file per (direction, supervision), patterns x datasets, written once
    # at the end -- the layout of the synthetic_tree_*.csv files, which
    # notebooks/VisualisationResults.ipynb reads beside it. It replaces the
    # appended results_performance_<direction>_<supervision>.txt log, whose
    # four-element lists had no room for the ranking metrics.
    out_path = RESULTS_DIR+'/synthetic_score_'+str_directed+'_'+str_supervised+'.csv'
    write_csv(pd.DataFrame(results_dict), out_path, index=True)
    print("wrote "+out_path)

if __name__ == "__main__":
    # GARGAML_DATASET=synthetic runs the 66-dataset synthetic grid instead, and
    # GARGAML_DATASET=louvain_sweep the Louvain arms (slurm/distribution_scores.slurm
    # takes either as its argument).
    selected = select_datasets(["HI-Small", "LI-Large"])  # synthetic, louvain_sweep, HI-Small, LI-Large
    datasets = []
    for name in selected:
        datasets += LOUVAIN_SWEEP if name == "louvain_sweep" else [name]
    for dataset in datasets:
        for directed in [True, False]:
            supervised = True
            score_type = "weighted_average" # basic or weighted_average

            if dataset == "synthetic":
                benchmark_synthetic(directed, supervised, score_type)
            else: 
                general_calculation(dataset, directed, supervised, score_type)
