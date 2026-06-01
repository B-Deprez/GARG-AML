import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from sklearn.metrics import precision_score, f1_score, roc_auc_score, average_precision_score

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.methods.gargaml_scores import define_gargaml_scores
import pandas as pd
import timeit

import matplotlib.pyplot as plt
import numpy as np

def divergence_metric(dist_0, dist_1):
    mean_0 = np.mean(dist_0)
    variance_0 = np.var(dist_0)
    mean_1 = np.mean(dist_1)
    variance_1 = np.var(dist_1)
    return (mean_0 - mean_1)**2 + 0.5*(variance_0 + variance_1)

def lift_curve_values(y_val, y_pred, steps):
    vals_lift = [] #The lift values to be plotted

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

    cut_offs = [0.1, 0.2, 0.3, 0.5, 0.9]
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
        
            # Filter the DataFrame by label
            label_0 = results_df[results_df["Label"] == 0]["GARGAML"]
            label_1 = results_df[results_df["Label"] == 1]["GARGAML"]

            # Calculate the bin edges
            all_data = np.concatenate([label_0, label_1])
            bins = np.histogram_bin_edges(all_data, bins=20)

            # Plot histogram for label 0
            axes[i, j].hist(label_0, bins=bins, alpha=0.5, label='Label 0', density=True)

            # Plot histogram for label 1
            axes[i, j].hist(label_1, bins=bins, alpha=0.5, label='Label 1', density=True)

            divergence = divergence_metric(label_0, label_1)
            divergence_matrix[i, j] = divergence

    divergence_df = pd.DataFrame(divergence_matrix, columns=columns, index=cut_offs)
    divergence_df.to_csv("results-0/"+dataset+"_GARGAML_"+str_supervised+"_"+str_directed+"_combined_divergence.csv")

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
    plt.savefig("results-0/"+dataset+"_GARGAML_"+str_supervised+"_"+str_directed+"_combined_histogram.pdf")
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
    plt.savefig("results-0/"+dataset+"_GARGAML_"+str_supervised+"_"+str_directed+"_combined_lift.pdf")
    plt.close()

    print("="*10)
    print("Figure lift saved")

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

    y_pred = labels_gargaml_full["GARGAML"].values

    cut_offs = [0.1, 0.2, 0.3, 0.5, 0.9]
    columns = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

    n = len(cut_offs)
    m = len(columns)

    for i in range(n):
        cut_off = cut_offs[i]
        for j in range(m):
            column = columns[j]
            y_true = ((labels_gargaml_full[column]>cut_off)*1).values
            precision = precision_score(y_true, (y_pred>0.5)*1)
            f1 = f1_score(y_true, (y_pred>0.5)*1)
            auc_roc = roc_auc_score(y_true, y_pred)
            auc_pr = average_precision_score(y_true, y_pred)

            with open('results/results_performance_IBM_'+str_directed+'.txt', 'a') as f:
                f.write(dataset+'_'+column+'_'+str(cut_off)+' [precision, F1, AUC-ROC, AUC-PR]: '+str([precision, f1, auc_roc, auc_pr])+'\n')

def plot_distribution_synthetic(laundering_combined, columns, str_directed, str_supervised):
    n = len(columns)

    fig, axes = plt.subplots(n//2, n//2+n%2, figsize = (3*n, 1.5*n))

    for i in range(n):
        # Distributions

        column = columns[i]
        laundering_combined["Label"] = laundering_combined[column].values

        # Filter the DataFrame by label
        label_0 = laundering_combined[laundering_combined["Label"] == 0]["GARGAML"]
        label_1 = laundering_combined[laundering_combined["Label"] == 1]["GARGAML"]

        # Calculate the bin edges
        all_data = np.concatenate([label_0, label_1])
        bins = np.histogram_bin_edges(all_data, bins=20)

        # Plot histogram for label 0
        axes[i//2, i%2].hist(label_0, bins=bins, alpha=0.5, label='Other', density=True)

        # Plot histogram for label 1
        axes[i//2, i%2].hist(label_1, bins=bins, alpha=0.5, label=column, density=True)

        # Add labels and title
        axes[i//2, i%2].legend()
        axes[i//2, i%2].set_xlabel('GARG-AML score')
        axes[i//2, i%2].set_ylabel('Relative Frequency')
        axes[i//2, i%2].set_title(column)
    fig.tight_layout()
    plt.savefig("results-0/synthetic_GARGAML_"+str_supervised+"_"+str_directed+"_histogram.pdf")
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
    plt.savefig("results-0/synthetic_GARGAML_"+str_supervised+"_"+str_directed+"_lift.pdf")
    plt.close()


def distribution_scores_synthetic(dataset, results_df, str_directed, str_supervised):
    columns = ['laundering', 'separate', 'new_mules', 'existing_mules']
    label_data = pd.read_csv("data/label_data_"+dataset+".csv")
    laundering_combined = results_df.merge(label_data, left_index=True, right_index=True, how="outer")
    laundering_combined.fillna(-1, inplace=True) # Nodes without connections: not smurfing according to us

    plot_distribution_synthetic(laundering_combined, columns, str_directed, str_supervised)

    plot_lift_synthetic(laundering_combined, columns, str_directed, str_supervised)


    results = dict()
    for column in columns:
        print(column)
        precision = precision_score(laundering_combined[column], (laundering_combined["GARGAML"]>0.5)*1)
        f1 = f1_score(laundering_combined[column], (laundering_combined["GARGAML"]>0.5)*1)
        auc_roc = roc_auc_score(laundering_combined[column], laundering_combined["GARGAML"])
        auc_pr = average_precision_score(laundering_combined[column], laundering_combined["GARGAML"])
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
        results_df_measures = pd.read_csv("results-3/"+dataset+"_GARGAML_"+str_directed+".csv")
        results_df = define_gargaml_scores(results_df_measures, directed=directed, score_type=score_type)

    else:
        results_df = pd.read_csv("results-3/"+dataset+"_GARGAML_"+str_directed+"_IF.csv")
        results_df = results_df.set_index("node")
        results_df = results_df[["anomaly_score"]]
        results_df["anomaly_score"] = results_df["anomaly_score"]*(-1)
        results_df.columns = ["GARGAML"]

    if dataset in ["HI-Small", "LI-Large"]:
        distribution_scores_IBM(dataset, results_df, str_directed, str_supervised)

    elif dataset[:min(9, len(dataset))] == "synthetic": #use min in case string would be shorter than 9. We don't want an error here
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

    n_nodes_list = [100, 10000, 100000] # Number of nodes in the graph
    m_edges_list = [1, 2, 5] # Number of edges to attach from a new node to existing nodes
    p_edges_list = [0.001, 0.01] # Probability of adding an edge between two nodes
    generation_method_list = [
        'Barabasi-Albert', 
        'Erdos-Renyi', 
        'Watts-Strogatz'
        ] # Generation method for the graph
    n_patterns_list = [3, 5] # Number of smurfing patterns to add

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
                            with open('results-0/results_performance_'+str_directed+'_'+str_supervised+'.txt', 'a') as f:
                                f.write(string_name+' [Precision, F1, AUC-ROC, AUC-PR]: '+str(results_int)+'\n')
                    if generation_method == 'Erdos-Renyi':
                        m_edges = 0
                        for p_edges in p_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            print("====", string_name, "====")
                            results_int = general_calculation(string_name, directed, supervised, score_type)
                            with open('results-0/results_performance_'+str_directed+'_'+str_supervised+'.txt', 'a') as f:
                                f.write(string_name+' [Precision, F1, AUC-ROC, AUC-PR]: '+str(results_int)+'\n')

                    if generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                                print("====", string_name, "====")
                                results_int = general_calculation(string_name, directed, supervised, score_type)
                                with open('results-0/results_performance_'+str_directed+'_'+str_supervised+'.txt', 'a') as f:
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
