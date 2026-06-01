from sklearn import tree

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels, combine_patterns_GARGAML
from src.methods.gargaml_scores import define_gargaml_scores, summarise_gargaml_scores
from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community

from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, f1_score, roc_auc_score, average_precision_score

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

def evaluate_model(clf, X_test, y_test, plot=False):
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    AUC_ROC = roc_auc_score(y_test, y_pred)
    AUC_PR = average_precision_score(y_test, y_pred)


    if plot:
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)

        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")

        plt.subplot(2, 1, 2)
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.show()


    return precision, f1, AUC_ROC, AUC_PR

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

def data_split(laundering_combined, gargaml_columns, target, cutoff):
    X_df = laundering_combined[gargaml_columns]
    rel_labels = laundering_combined[target]
    y = (rel_labels>cutoff)*1

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=1997, stratify=y)

    return X_train, X_test, y_train, y_test

def main():
    dataset = "HI-Small"  
    directed = True
    str_directed = "directed" if directed else "undirected"
    score_type = "weighted_average"

    cut_offs = [0.1, 0.2, 0.3, 0.5, 0.9]
    columns = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

    n = len(cut_offs)
    m = len(columns)

    precision_tree_matrix = np.zeros((n, m))
    precision_boosting_matrix = np.zeros((n, m))

    f1_tree_matrix = np.zeros((n, m))
    f1_boosting_matrix = np.zeros((n, m))

    AUC_ROC_tree_matrix = np.zeros((n, m))
    AUC_ROC_boosting_matrix = np.zeros((n, m))

    AUC_PR_tree_matrix = np.zeros((n, m))
    AUC_PR_boosting_matrix = np.zeros((n, m))

    imbalance_matrix = np.zeros((n,m))

    gargaml_columns = [
        "GARGAML", 
        "GARGAML_min", "GARGAML_max", "GARGAML_mean", "GARGAML_std",
        "degree", "degree_min", "degree_max", "degree_mean", "degree_std"
        ]

    print("Data preparation")
    laundering_combined = data_preparation(dataset, gargaml_columns, directed, score_type)

    for i in range(n):
        cutoff = cut_offs[i]
        for j in range(m):
            target = columns[j]

            print(cutoff, target)

            try: # If too few labels, the model will not work. Performance matrix will be filled with NaNs
                X_train, X_test, y_train, y_test = data_split(laundering_combined, gargaml_columns, target, cutoff)

                imbalance_matrix[i, j] = sum(y_train)/len(y_train)

                tree_clf = gargaml_tree(X_train, y_train)

                precision_tree, f1_tree, AUC_ROC_tree, AUC_PR_tree = evaluate_model(tree_clf, X_test, y_test)

                boosting_clf = gargaml_boosting(X_train, y_train)

                precision_boosting, f1_boosting, AUC_ROC_boosting, AUC_PR_boosting = evaluate_model(boosting_clf, X_test, y_test)

                precision_tree_matrix[i, j] = precision_tree
                f1_tree_matrix[i, j] = f1_tree
                AUC_ROC_tree_matrix[i, j] = AUC_ROC_tree
                AUC_PR_tree_matrix[i, j] = AUC_PR_tree

                precision_boosting_matrix[i, j] = precision_boosting
                f1_boosting_matrix[i, j] = f1_boosting
                AUC_ROC_boosting_matrix[i, j] = AUC_ROC_boosting
                AUC_PR_boosting_matrix[i, j] = AUC_PR_boosting

            except:
                precision_tree_matrix[i, j] = np.nan
                f1_tree_matrix[i, j] = np.nan
                AUC_ROC_tree_matrix[i, j] = np.nan
                AUC_PR_tree_matrix[i, j] = np.nan

                precision_boosting_matrix[i, j] = np.nan
                f1_boosting_matrix[i, j] = np.nan
                AUC_ROC_boosting_matrix[i, j] = np.nan
                AUC_PR_boosting_matrix[i, j] = np.nan
    
    precision_tree_df = pd.DataFrame(precision_tree_matrix, columns=columns, index=cut_offs)
    precision_tree_df.to_csv("results/"+dataset+"_precision_tree_"+str_directed+"_combined.csv")
    precision_boosting_df = pd.DataFrame(precision_boosting_matrix, columns=columns, index=cut_offs)
    precision_boosting_df.to_csv("results/"+dataset+"_precision_boosting_"+str_directed+"_combined.csv")
    f1_tree_df = pd.DataFrame(f1_tree_matrix, columns=columns, index=cut_offs)
    f1_tree_df.to_csv("results/"+dataset+"_f1_tree_"+str_directed+"_combined.csv")
    f1_boosting_df = pd.DataFrame(f1_boosting_matrix, columns=columns, index=cut_offs)
    f1_boosting_df.to_csv("results/"+dataset+"_f1_boosting_"+str_directed+"_combined.csv")

    AUC_ROC_tree_df = pd.DataFrame(AUC_ROC_tree_matrix, columns=columns, index=cut_offs)
    AUC_ROC_tree_df.to_csv("results/"+dataset+"_AUC_ROC_tree_"+str_directed+"_combined.csv")
    AUC_ROC_boosting_df = pd.DataFrame(AUC_ROC_boosting_matrix, columns=columns, index=cut_offs)
    AUC_ROC_boosting_df.to_csv("results/"+dataset+"_AUC_ROC_boosting_"+str_directed+"_combined.csv")
    AUC_PR_tree_df = pd.DataFrame(AUC_PR_tree_matrix, columns=columns, index=cut_offs)
    AUC_PR_tree_df.to_csv("results/"+dataset+"_AUC_PR_tree_"+str_directed+"_combined.csv")
    AUC_PR_boosting_df = pd.DataFrame(AUC_PR_boosting_matrix, columns=columns, index=cut_offs)
    AUC_PR_boosting_df.to_csv("results/"+dataset+"_AUC_PR_boosting_"+str_directed+"_combined.csv")
    imbalance_df = pd.DataFrame(imbalance_matrix, columns=columns, index=cut_offs)
    imbalance_df.to_csv("results/"+dataset+"_imbalance_"+str_directed+"_combined.csv")

if __name__ == "__main__":
    main()