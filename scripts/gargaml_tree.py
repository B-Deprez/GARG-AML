from sklearn import tree

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels, combine_patterns_GARGAML
from src.methods.gargaml_scores import define_gargaml_scores, summarise_gargaml_scores
from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community
from src.utils.evaluation import (
    SEED,
    evaluate_model,
    holdout_split,
    metric_records,
    nan_metrics,
    write_metrics,
)
from src.utils.naming import gargaml_key

from sklearn import tree
from sklearn import ensemble

from pickle import dump

def gargaml_tree(X, y, save = False, save_path = "results/model_tree.pkl"):
    clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
    clf = clf.fit(X, y)

    if save:
        with open(save_path, "wb") as f:
            dump(clf, f, protocol=5)

    return clf

def gargaml_boosting(X, y, save = False, save_path = "results/model_boosting.pkl"):
    clf = ensemble.GradientBoostingClassifier(min_samples_leaf=10, random_state=1997)
    clf = clf.fit(X, y)

    if save:
        with open(save_path, "wb") as f:
            dump(clf, f, protocol=5)

    return clf

def data_preparation(dataset, gargaml_columns, directed, score_type):
    str_directed = "directed" if directed else "undirected"

    results_df_measures = pd.read_csv("results/"+dataset+"_GARGAML_"+str_directed+".csv") #measures

    results_df = define_gargaml_scores(results_df_measures, directed, score_type=score_type) #summary scores

    transactions_df_extended, pattern_columns = define_ML_labels( #patterns
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    path = "data/"+dataset+"_Trans.csv"
    G = construct_IBM_graph(path=path, directed = False) #For summary, we use the undirected graph
    G_reduced = graph_community(G)

    summary_gargaml = summarise_gargaml_scores(G_reduced, results_df, columns = gargaml_columns)
    for column in gargaml_columns:
        results_df[column] = summary_gargaml[column]

    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)

    combined_patterns_GARGAML = combine_patterns_GARGAML(results_df, laundering_combined, columns = gargaml_columns)

    for column in gargaml_columns:
        laundering_combined[column] = combined_patterns_GARGAML[column]

    return laundering_combined

def data_split(laundering_combined, gargaml_columns, target, cutoff, seed=SEED):
    X_df = laundering_combined[gargaml_columns]
    rel_labels = laundering_combined[target]
    y = (rel_labels>cutoff)*1

    X_train, X_test, y_train, y_test = holdout_split(X_df, y, seed=seed)

    return X_train, X_test, y_train, y_test

def main():
    dataset = "HI-Small"  
    directed = True
    str_directed = "directed" if directed else "undirected"
    score_type = "weighted_average"
    seed = SEED

    cut_offs = [0.1, 0.2, 0.3, 0.5, 0.9]
    columns = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

    gargaml_columns = [
        "GARGAML", 
        "GARGAML_min", "GARGAML_max", "GARGAML_mean", "GARGAML_std",
        "degree", "degree_min", "degree_max", "degree_mean", "degree_std"
        ]

    models = [
        (gargaml_key("tree", directed), gargaml_tree),
        (gargaml_key("boost", directed), gargaml_boosting),
    ]

    print("Data preparation")
    laundering_combined = data_preparation(dataset, gargaml_columns, directed, score_type)

    records = []

    for cutoff in cut_offs:
        for target in columns:
            print(cutoff, target)

            context = dict(
                dataset = dataset,
                direction = str_directed,
                features = "full", # this script's existing feature set: GARG-AML score (a)
                                    # + neighbour degree stats (c) + neighbour score stats (d).
                                    # No raw block density/size columns (b) -- gargaml_columns
                                    # above never carried them, gargaml_tree_blocks.py isolates
                                    # them separately. "full" means "the model as historically
                                    # trained here", not literally all four task-3 groups; task 3's
                                    # refactor decides whether this tag/column-set changes.
                cutoff = cutoff,
                target = target,
                seed = seed,
            )

            try:
                X_train, X_test, y_train, y_test = data_split(laundering_combined, gargaml_columns, target, cutoff, seed=seed)
            except Exception as exc: # Too few labels to even split: no models for this cell
                print("    no split: "+repr(exc))
                records += metric_records({"imbalance": 0.0}, status = "skipped: "+str(exc), model = "", **context)
                for model_key, _ in models:
                    records += metric_records(nan_metrics(), status = "skipped: "+str(exc), model = model_key, **context)
                continue

            n_test = len(y_test)
            n_pos = int(sum(y_test))

            records += metric_records(
                {"imbalance": sum(y_train)/len(y_train)},
                n_test = n_test, n_pos = n_pos, model = "", **context
                )

            for model_key, fit_model in models:
                try: # If too few labels, the model will not work. The cell is reported as NaN, with the reason
                    clf = fit_model(X_train, y_train)
                    metrics = evaluate_model(clf, X_test, y_test)
                    status = "ok"
                except Exception as exc:
                    print("    "+model_key+" skipped: "+repr(exc))
                    metrics = nan_metrics()
                    status = "skipped: "+str(exc)

                records += metric_records(
                    metrics, status = status, n_test = n_test, n_pos = n_pos,
                    model = model_key, **context
                    )

    write_metrics(records, dataset, str_directed)

if __name__ == "__main__":
    main()