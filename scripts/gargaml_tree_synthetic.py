from sklearn import tree

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)


import numpy as np
import pandas as pd

from src.methods.gargaml_scores import define_gargaml_scores, summarise_gargaml_scores
from src.data.graph_construction import construct_synthetic_graph
from src.utils.graph_processing import graph_community
from src.utils.evaluation import SEED, evaluate_model, holdout_split, metric_names
from src.utils.runtime import (as_bool, echo_config, env_override,
                               resolve_results_dir, write_csv)

from sklearn import tree
from sklearn import ensemble

RESULTS_DIR = resolve_results_dir()

# Result cells are repr()'d dicts parsed back with a bare eval() in
# notebooks/VisualisationResults.ipynb, which has no `nan` name in scope -- a
# stray NaN anywhere raises there and zeroes out the whole cell's metrics. So
# -1 stands for "missing" (data_preparation's fillna(-1) below) and 0 for a
# total training failure. LEGACY_METRIC_RENAME maps the shared module's key
# spelling to this file's capitalisation; _legacy_metric_dict substitutes -1
# for any NaN (only possible from R@K/lift@K on zero positives) so every cell
# stays eval()-safe.
LEGACY_METRIC_RENAME = {"precision": "Precision", "f1": "F1"}


def _legacy_metric_dict(metrics):
    out = {}
    for key, value in metrics.items():
        key = LEGACY_METRIC_RENAME.get(key, key)
        if isinstance(value, float) and np.isnan(value):
            value = -1
        out[key] = value
    return out


def _legacy_zero_metrics():
    return {LEGACY_METRIC_RENAME.get(k, k): 0 for k in metric_names()}

def data_preparation(dataset, gargaml_columns, directed, score_type):
    directed_str = 'directed' if directed else 'undirected'
    # The undirected measures are written with a "_parallel" suffix by
    # gargaml_undirected_synth.py; directed is not.
    if directed:
        path_res = RESULTS_DIR+'/'+dataset+'_GARGAML_'+directed_str+'.csv'
    else:
        path_res = RESULTS_DIR+'/'+dataset+'_GARGAML_'+directed_str+'_parallel.csv'
    results_df_measures = pd.read_csv(path_res)
    results_df = define_gargaml_scores(results_df_measures, directed=directed, score_type=score_type)
    path = 'data/edge_data_'+dataset+'.csv'
    G = construct_synthetic_graph(path=path, directed = directed)
    G_reduced = graph_community(G)
    summary_gargaml = summarise_gargaml_scores(G_reduced, results_df, columns = gargaml_columns)

    for column in gargaml_columns:
        results_df[column] = summary_gargaml[column]
    return merge_labels(results_df, dataset)

def merge_labels(results_df, dataset):
    """The node labels joined onto a score frame, in the row order the split sees.

    Shared with scripts/distribution_scores.py, which builds the base score's
    frame through it so that :func:`holdout_test_index` hands back the same
    test rows there as here.
    """
    path = 'data/label_data_'+dataset+'.csv'
    labels_df = pd.read_csv(path)
    labels_df.reset_index(inplace=True)
    labels_df = labels_df.rename(columns={"index": "node"})

    results_df = results_df.merge(labels_df, on='node', how='outer')
    results_df.fillna(-1, inplace=True)
    return results_df

def data_split(results_df, gargaml_columns, target, test_size=0.3, seed=SEED):
    X_df = results_df[gargaml_columns]
    y = results_df[target]*1

    X_train, X_test, y_train, y_test = holdout_split(X_df, y, test_size=test_size, seed=seed)

    return X_train, X_test, y_train, y_test

def holdout_test_index(results_df, target, test_size=0.3, seed=SEED):
    """Row labels :func:`data_split` puts in the test set for ``target``.

    The split depends on the label column and the row order only, not on the
    features, so the base score can be scored on exactly the rows the tree
    models are tested on.
    """
    _, X_test, _, _ = data_split(results_df, [target], target, test_size=test_size, seed=seed)
    return X_test.index

GARGAML_COLUMNS = [
    "GARGAML", 
    "GARGAML_min", "GARGAML_max", "GARGAML_mean", "GARGAML_std",
    "degree", "degree_min", "degree_max", "degree_mean", "degree_std"
    ]

def train_pipeline(data_tree, string_name, pattern, tree_model):
    print("====================================")
    print(string_name)
    print(pattern)
    print(tree_model)
    X_train, X_test, y_train, y_test = data_split(data_tree, GARGAML_COLUMNS, target=pattern, test_size=0.3)

    if tree_model == 'tree':
        clf = tree.DecisionTreeClassifier(min_samples_leaf=10, random_state=1997)
        clf.fit(X_train, y_train)
    elif tree_model == 'boosting':
        clf = ensemble.GradientBoostingClassifier(min_samples_leaf=10, random_state=1997)
        clf.fit(X_train, y_train)
    else:   
        raise ValueError("Invalid tree model specified. Choose 'tree' or 'boosting'.")

    return _legacy_metric_dict(evaluate_model(clf, X_test, y_test))

def gargaml_tree_synthetic(string_name, directed):
    patterns = [
        'laundering',
        'separate',
        'new_mules', 
        'existing_mules',
    ]
    tree_models = [
        'tree',
        'boosting'
    ]
    # Prepared once per dataset rather than once per (pattern, model): none
    # of it depends on either, and it rebuilds the graph and reruns Louvain
    # (seeded, so the result is the same each time) -- ~160 s per call on
    # the densest 100,000-node datasets, eight calls per dataset before.
    # A failure here zeroes every cell, as it did when it ran per cell.
    try:
        data_tree = data_preparation(string_name, GARGAML_COLUMNS, directed, score_type='weighted_average')
    except Exception as exc:
        print("Error in data preparation for: {} ({!r})".format(string_name, exc))
        return {pattern: {tree_model: _legacy_zero_metrics() for tree_model in tree_models}
                for pattern in patterns}

    results = {}
    for pattern in patterns:
        results[pattern] = {}
        for tree_model in tree_models:
            try:
                metrics = train_pipeline(data_tree, string_name, pattern, tree_model)
            except Exception as exc:
                print("Error in training pipeline for: {} ({!r})".format(string_name, exc))
                metrics = _legacy_zero_metrics()
            results[pattern][tree_model] = metrics
    return results


def main():
    # GARGAML_DIRECTED=1 selects the directed measures; a bare run keeps the
    # undirected default. Each direction is its own output file, so the two
    # are separate runs (slurm/tree_synth.slurm takes it in its argument).
    directed = env_override("directed", False, as_bool)
    echo_config(os.path.basename(__file__), directed=directed, results_dir=RESULTS_DIR)
    n_nodes_list = [
        100, 
        10000, 
        100000
        ]
    
    m_edges_list = [
        1, 
        2, 
        5
        ] # BA: edges attached from a new node to existing nodes
    
    p_edges_list = [
        0.001, 
        0.01
        ] # WS: rewiring probability
    
    generation_method_list = [
        'Barabasi-Albert', 
        'Erdos-Renyi', 
        'Watts-Strogatz'
        ]
    
    n_patterns_list = [
        3, 
        5
        ]

    results_dict = {}
    for n_nodes in n_nodes_list:
        for n_patterns in n_patterns_list:
            if n_patterns <= 0.06*n_nodes:
                for generation_method in generation_method_list:
                    if generation_method == 'Barabasi-Albert':
                        p_edges = 0
                        for m_edges in m_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            results = gargaml_tree_synthetic(string_name, directed)
                            results_dict[string_name] = results
                    if generation_method == 'Erdos-Renyi':
                        m_edges = 0
                        for p_edges in p_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            results = gargaml_tree_synthetic(string_name, directed)
                            results_dict[string_name] = results
                    if generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                                results = gargaml_tree_synthetic(string_name, directed)
                                results_dict[string_name] = results
    results_df = pd.DataFrame(results_dict)
    # Into the results directory, beside the measures it was computed from --
    # not the repository root, where the published run's files live.
    write_csv(results_df, RESULTS_DIR+"/synthetic_tree_"+str(directed)+"_full.csv", index=True)

if __name__ == '__main__':
    main()