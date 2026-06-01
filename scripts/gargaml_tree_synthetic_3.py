from sklearn import tree

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)


import pandas as pd
import matplotlib.pyplot as plt

from src.methods.gargaml_scores import define_gargaml_scores, summarise_gargaml_scores
from src.data.graph_construction import construct_synthetic_graph
from src.utils.graph_processing import graph_community

from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, f1_score, roc_auc_score, average_precision_score

def data_preparation(dataset, gargaml_columns, directed, score_type):
    directed_str = 'directed' if directed else 'undirected'
    # Load the dataset
    path_res = 'results-0/'+dataset+'_GARGAML_'+directed_str+'.csv'
    results_df_measures = pd.read_csv(path_res)
    # Define GARG-AML scores
    results_df = define_gargaml_scores(results_df_measures, directed=directed, score_type=score_type)
    # Summarise GARG-AML scores
    path = 'data/edge_data_'+dataset+'.csv'
    G = construct_synthetic_graph(path=path, directed = directed)
    G_reduced = graph_community(G)
    summary_gargaml = summarise_gargaml_scores(G_reduced, results_df, columns = gargaml_columns)

    for column in gargaml_columns:
        results_df[column] = summary_gargaml[column]
    # Load ML labels
    path = 'data/label_data_'+dataset+'.csv'
    labels_df = pd.read_csv(path)
    labels_df.reset_index(inplace=True)
    labels_df = labels_df.rename(columns={"index": "node"})

    # Combine labels with GARG-AML scores
    results_df = results_df.merge(labels_df, on='node', how='outer')
    results_df.fillna(-1, inplace=True)
    return results_df

def data_split(results_df, gargaml_columns, target, test_size=0.3):
    X_df = results_df[gargaml_columns]
    y = results_df[target]*1

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=1997, stratify=y)

    return X_train, X_test, y_train, y_test

def train_pipeline(string_name, pattern, tree_model, directed):
    gargaml_columns = [
        "GARGAML", 
        "GARGAML_min", "GARGAML_max", "GARGAML_mean", "GARGAML_std",
        "degree", "degree_min", "degree_max", "degree_mean", "degree_std"
        ]
    
    print("====================================")
    print(string_name)
    print(pattern)
    print(tree_model)
    data_tree = data_preparation(string_name, gargaml_columns, directed, score_type='weighted_average')
    X_train, X_test, y_train, y_test = data_split(data_tree, gargaml_columns, target=pattern, test_size=0.3)

    # Train the model
    if tree_model == 'tree':
        clf = tree.DecisionTreeClassifier(min_samples_leaf=10, random_state=1997)
        clf.fit(X_train, y_train)
        #tree.plot_tree(
        #    clf, 
        #    feature_names=gargaml_columns, 
        #    proportion=False,
        #    impurity=False, 
        #    class_names=['not_'+pattern, pattern],
        #    filled=True
        #    )
        #plt.savefig('res/tree_'+string_name+'_'+pattern+'.pdf')
    elif tree_model == 'boosting':
        clf = ensemble.GradientBoostingClassifier(min_samples_leaf=10, random_state=1997)
        clf.fit(X_train, y_train)
    else:   
        raise ValueError("Invalid tree model specified. Choose 'tree' or 'boosting'.")

    # Evaluate model 
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    AUC_ROC = roc_auc_score(y_test, y_pred)
    AUC_PR = average_precision_score(y_test, y_pred)
    return precision, f1, AUC_ROC, AUC_PR

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
    results = {}
    for pattern in patterns:
        results[pattern] = {}
        for tree_model in tree_models:
            try:
                precision, f1, AUC_ROC, AUC_PR = train_pipeline(string_name, pattern, tree_model, directed)
            except:
                print("Error in training pipeline for: {}".format(string_name))
                precision = f1 = AUC_ROC = AUC_PR = 0
            results[pattern][tree_model] = {
                'Precision': precision,
                'F1': f1,
                'AUC_ROC': AUC_ROC,
                'AUC_PR': AUC_PR
            }
    return results


def main():
    directed = False
    n_nodes_list = [
        100, 
        10000, 
        100000
        ] # Number of nodes in the graph
    
    m_edges_list = [
        1, 
        2, 
        5
        ] # Number of edges to attach from a new node to existing nodes
    
    p_edges_list = [
        0.001, 
        0.01
        ] # Probability of adding an edge between two nodes
    
    generation_method_list = [
        'Barabasi-Albert', 
        'Erdos-Renyi', 
        'Watts-Strogatz'
        ] # Generation method for the graph
    
    n_patterns_list = [
        3, 
        #5
        ] # Number of smurfing patterns to add

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
    # Save results
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("synthetic_tree_"+str(directed)+"_3.csv")

if __name__ == '__main__':
    main()