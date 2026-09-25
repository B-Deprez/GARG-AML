import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.methods.gargaml_scores import define_gargaml_scores
from src.utils.evaluation import (CUT_OFFS, LEGACY_METRICS, SEED,
                                  evaluate_scores, fold_assignments, folds_path,
                                  metric_names, metric_records, nan_metrics,
                                  read_folds, write_metrics)
from src.utils.naming import gargaml_key
from src.utils.runtime import resolve_results_dir
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

# This file's convention for "not available" is -1, not NaN: results are
# logged as repr()'d Python literals and read back with a bare eval() in
# notebooks/VisualisationResults.ipynb, which has no `nan` name in scope.
# _sanitise substitutes -1 before anything gets str()'d into a log line.
def _sanitise(value):
    return -1 if isinstance(value, float) and np.isnan(value) else value

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


def distribution_scores_IBM(dataset, results_df, str_directed, str_supervised):
    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)
    del transactions_df_extended
    del pattern_columns

    labels_gargaml_full = laundering_combined.merge(results_df[["GARGAML"]], left_index=True, right_index=True, how="outer").fillna(-1)
    del laundering_combined

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


def distribution_scores_synthetic(dataset, results_df, str_directed, str_supervised):
    """Base-score metrics on one synthetic dataset, un-folded.

    Cross-validation is scoped to the IBM data: the 66 synthetic datasets
    keep the single split, and their variance comes from the 66-dataset
    spread that feeds the Friedman/Nemenyi analysis, so evaluation here is
    over the full population.

    The returned list length is load-bearing and must stay 4 (see the
    comment on the ``results[column]`` assignment below).
    """
    columns = ['laundering', 'separate', 'new_mules', 'existing_mules']
    label_data = pd.read_csv("data/label_data_"+dataset+".csv")
    laundering_combined = results_df.merge(label_data, left_index=True, right_index=True, how="outer")
    laundering_combined.fillna(-1, inplace=True) # Nodes without connections are not smurfing

    plot_distribution_synthetic(laundering_combined, columns, str_directed, str_supervised)

    plot_lift_synthetic(laundering_combined, columns, str_directed, str_supervised)


    results = dict()
    for column in columns:
        print(column)
        y_true = laundering_combined[column].values
        y_pred = laundering_combined["GARGAML"].values

        try:
            # No natural 0/1 prediction for a raw score -- see distribution_scores_IBM.
            metrics = evaluate_scores(y_true, y_pred)
            precision, f1, auc_roc, auc_pr = (_sanitise(metrics[k]) for k in LEGACY_METRICS)
        except Exception as exc:
            print("    skipped: "+repr(exc))
            precision = f1 = auc_roc = auc_pr = -1

        # Fixed at exactly 4: notebooks/VisualisationResults.ipynb's
        # gargaml_results() unpacks this as `precision, f1_score, ROC, PR =
        # tuple(...)`, so a 5th element raises there on every line.
        results[column] = [precision, f1, auc_roc, auc_pr]
        print("Precision: ", precision)
        print("F1: ", f1)
        print("AUC-ROC: ", auc_roc)
        print("AUC-PR: ", auc_pr)

    return results

def general_calculation(dataset, directed, supervised, score_type):
    str_directed = "directed" if directed else "undirected"
    str_supervised = "supervised" if supervised else "unsupervised"

    if supervised:
        results_df_measures = pd.read_csv(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_directed+".csv")
        results_df = define_gargaml_scores(results_df_measures, directed=directed, score_type=score_type)

    else:
        results_df = pd.read_csv(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_directed+"_IF.csv")
        results_df = results_df.set_index("node")
        results_df = results_df[["anomaly_score"]]
        results_df["anomaly_score"] = results_df["anomaly_score"]*(-1)
        results_df.columns = ["GARGAML"]

    if dataset in ["HI-Small", "LI-Large"]:
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

    for n_nodes in n_nodes_list:
        for n_patterns in n_patterns_list:
            if n_patterns <= 0.06*n_nodes:
                for generation_method in generation_method_list:
                    if generation_method == 'Barabasi-Albert':
                        p_edges = 0
                        for m_edges in m_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            print("====", string_name, "====")
                            results_int = general_calculation(string_name, directed, supervised, score_type)
                            with open(RESULTS_DIR+'/results_performance_'+str_directed+'_'+str_supervised+'.txt', 'a') as f:
                                f.write(string_name+' [Precision, F1, AUC-ROC, AUC-PR]: '+str(results_int)+'\n')
                    if generation_method == 'Erdos-Renyi':
                        m_edges = 0
                        for p_edges in p_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            print("====", string_name, "====")
                            results_int = general_calculation(string_name, directed, supervised, score_type)
                            with open(RESULTS_DIR+'/results_performance_'+str_directed+'_'+str_supervised+'.txt', 'a') as f:
                                f.write(string_name+' [Precision, F1, AUC-ROC, AUC-PR]: '+str(results_int)+'\n')

                    if generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                                print("====", string_name, "====")
                                results_int = general_calculation(string_name, directed, supervised, score_type)
                                with open(RESULTS_DIR+'/results_performance_'+str_directed+'_'+str_supervised+'.txt', 'a') as f:
                                    f.write(string_name+' [Precision, F1, AUC-ROC, AUC-PR]: '+str(results_int)+'\n')

if __name__ == "__main__":
    datasets = ["HI-Small", "LI-Large"]  #Synthetic, HI-Small, LI-Large
    for dataset in datasets:
        for directed in [True, False]:
            supervised = True
            score_type = "weighted_average" # basic or weighted_average

            if dataset == "synthetic":
                benchmark_synthetic(directed, supervised, score_type)
            else: 
                general_calculation(dataset, directed, supervised, score_type)
