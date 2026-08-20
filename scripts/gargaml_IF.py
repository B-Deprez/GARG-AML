# LOAD MODULES
import os
import sys
import itertools
import warnings
from tqdm import tqdm

import networkx as nx
import pandas as pd

from sklearn.ensemble import IsolationForest

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

from src.methods.utils.measure_functions_directed import *
from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community
from src.methods.utils.neighbourhood_functions import GARG_AML_nodeselection

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels
from src.utils.evaluation import SEED, evaluate_scores, metric_records, nan_metrics, write_metrics

# IsolationForest is unsupervised and never sees labels during fit, so fitting it on
# every account (rather than a 70/30 split) is not label leakage -- it is standard
# practice for a density estimator, and keeps this baseline's published numbers
# unchanged from earlier revisions. Unlike gargaml_tree.py/_blocks.py/_synthetic*.py,
# this script's own AUC computation already used the continuous anomaly score, not
# clf.predict() -- it was never one of the five call sites with the known defect.
IF_MODEL_KEY = "gargaml_if_d"  # directed only; see src/utils/naming.py

def gargaml_IF(dataset = "HI-Small", directed = True):
    path = "data/"+dataset+"_Trans.csv"

    str_directed = "directed" if directed else "undirected"
    result_path = "results/"+dataset+"_GARGAML_"+str_directed+".csv"

    if os.path.exists(result_path):
        measure_df = pd.read_csv(result_path)
        measure_df = measure_df[[
            "node", 
            "measure_00", 
            "measure_01", 
            "measure_02", 
            "measure_10", 
            "measure_11", 
            "measure_12", 
            "measure_20", 
            "measure_21",
            "measure_22"
            ]]

    else: #If the results do not exist, calculate them
        G = construct_IBM_graph(path=path, directed = directed)
        G_reduced = graph_community(G)

        G_reduced_und = G_reduced.to_undirected()
        G_reduced_rev = G_reduced.reverse(copy=True)

        nodes = list(G_reduced.nodes)
        measure_00_list = []
        measure_01_list = []
        measure_02_list = []
        measure_10_list = []
        measure_11_list = []
        measure_12_list = []
        measure_20_list = []
        measure_21_list = []
        measure_22_list = []

        for node in tqdm(nodes):
            G_ego_second_und = nx.ego_graph(G_reduced_und, node, 2) #Use both incoming and outgoing edges
            G_ego_second = nx.subgraph(G_reduced, G_ego_second_und.nodes)
            G_ego_second_rev = nx.ego_graph(G_reduced_rev, node, 2) #Look at the reverse graph to get the incoming edges

            nodes_0, nodes_1, nodes_2, nodes_ordered = GARG_AML_nodeselection(G_ego_second, node, directed = True, G_ego_second_und = G_ego_second_und, G_ego_second_rev = G_ego_second_rev)

            adj_full = nx.adjacency_matrix(G_ego_second, nodelist=nodes_ordered).toarray()

            size_0 = len(nodes_0)
            size_1 = len(nodes_1)
            size_2 = len(nodes_2)

            measure_00 = measure_00_function(adj_full, size_0)
            measure_01 = measure_01_function(adj_full, size_0, size_1)
            measure_02 = measure_02_function(adj_full, size_0, size_1, size_2)
            measure_10 = measure_10_function(adj_full, size_0, size_1)
            measure_11 = measure_11_function(adj_full, size_0, size_1)
            measure_12 = measure_12_function(adj_full, size_0, size_1, size_2)  
            measure_20 = measure_20_function(adj_full, size_0, size_2)
            measure_21 = measure_21_function(adj_full, size_0, size_1, size_2)
            measure_22 = measure_22_function(adj_full, size_2)

            measure_00_list.append(measure_00)
            measure_01_list.append(measure_01)
            measure_02_list.append(measure_02)
            measure_10_list.append(measure_10)
            measure_11_list.append(measure_11)
            measure_12_list.append(measure_12)
            measure_20_list.append(measure_20)
            measure_21_list.append(measure_21)
            measure_22_list.append(measure_22)

        data_dict = {
            "node": nodes, 
            "measure_00": measure_00_list,
            "measure_01": measure_01_list, 
            "measure_02": measure_02_list,
            "measure_10": measure_10_list,
            "measure_11": measure_11_list,
            "measure_12": measure_12_list,
            "measure_20": measure_20_list,
            "measure_21": measure_21_list,
            "measure_22": measure_22_list,
                    }

        measure_df = pd.DataFrame(data_dict)

    X = measure_df.drop(columns = ["node"])

    clf = IsolationForest(random_state=1997)
    clf.fit(X)
    y_scores = clf.score_samples(X)

    measure_df["anomaly_score"] = y_scores

    return(measure_df)

def data_preparation(dataset, measure_df):
    anomaly_scores = measure_df.set_index("node")
    anomaly_scores = anomaly_scores[["anomaly_score"]]
    anomaly_scores['anomaly_score'] = -anomaly_scores['anomaly_score']

    transactions_df_extended, pattern_columns = define_ML_labels( #patterns
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)
    laundering_combined.drop(columns = ["Not Classified"], inplace = True)
    
    laundering_combined = laundering_combined.merge(anomaly_scores, left_index = True, right_index = True)
    
    return laundering_combined

def IF_AUC(dataset, measure_df, directed=True):
    str_directed = "directed" if directed else "undirected"
    cut_offs = [0.1, 0.2, 0.3, 0.5, 0.9]
    columns = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

    laundering_combined = data_preparation(dataset, measure_df)
    y_score_full = laundering_combined['anomaly_score']

    records = []
    for cutoff in cut_offs:
        for target in columns:
            print(cutoff, target)

            # Evaluated on every account, not a 30% slice: fit and score already ran
            # on the full population above, so this reports what was actually
            # computed, and keeps this baseline's published numbers unchanged.
            context = dict(
                dataset=dataset, direction=str_directed,
                features="block_densities_only",  # measure_00..measure_22 only, no
                                                    # sizes -- not literally task 3's
                                                    # group (b), which also has sizes
                cutoff=cutoff, target=target, seed=SEED,
            )
            y_true = (laundering_combined[target] > cutoff).astype(int)

            try:
                # No y_pred: this baseline has no natural 0/1 prediction, so
                # precision/f1 come back NaN rather than invented at some threshold
                # (evaluate_scores' documented behaviour) -- only AUC/ranking metrics
                # were ever meaningful for it.
                metrics = evaluate_scores(y_true, y_score_full)
                status = "ok"
            except Exception as exc:
                print("    skipped: "+repr(exc))
                metrics = nan_metrics()
                status = "skipped: "+str(exc)

            records += metric_records(
                metrics, status=status,
                n_test=len(y_true), n_pos=int(y_true.sum()),
                model=IF_MODEL_KEY, **context
            )

    write_metrics(records, dataset, str_directed, suffix="_if")

if __name__ == "__main__":
    dataset = "HI-Small"
    directed = True
    need_to_save = True
    str_directed = "directed" if directed else "undirected"
    measure_df = gargaml_IF(dataset = dataset, directed = directed)
    if need_to_save:
        measure_df.to_csv("results/"+dataset+"_GARGAML_"+str_directed+"_IF.csv", index = False)
    IF_AUC(dataset, measure_df, directed=directed)
