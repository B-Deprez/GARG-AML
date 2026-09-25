import numpy as np
import pandas as pd

from .utils.neighbourhood_functions import summaries_neighbourhoors_node, degree_neighbours_node, combine_GARG_AML

def calculate_score_directed(line, score_type="basic"):
    measure_00 = line["measure_00"]
    measure_01 = line["measure_01"]
    measure_02 = line["measure_02"]
    measure_10 = line["measure_10"]
    measure_11 = line["measure_11"]
    measure_12 = line["measure_12"]
    measure_20 = line["measure_20"]
    measure_21 = line["measure_21"]
    measure_22 = line["measure_22"]

    if score_type == "basic":
        measure_high = np.mean([measure_01, measure_12])
        measure_low = np.mean([measure_10, measure_21, measure_00, measure_02, measure_11, measure_20, measure_22])
        measure = measure_high - measure_low

        measure_high_transpose = np.mean([measure_10, measure_21])
        measure_low_transpose = np.mean([measure_01, measure_12, measure_00, measure_20, measure_11, measure_02, measure_22])
        measure_transpose = measure_high_transpose - measure_low_transpose

    elif score_type == "weighted_average":
        size_00 = line["size_00"]
        size_01 = line["size_01"]
        size_02 = line["size_02"]
        size_10 = line["size_10"]
        size_11 = line["size_11"]
        size_12 = line["size_12"]
        size_20 = line["size_20"]
        size_21 = line["size_21"]
        size_22 = line["size_22"]

        if size_01 + size_12 > 0:
            measure_high = (size_01*measure_01 + size_12*measure_12)/(size_01 + size_12)
        else:
            measure_high = np.mean([measure_01, measure_12])  # both sizes 0: revert to basic measure

        if size_10 + size_21 + size_00 + size_02 + size_11 + size_20 + size_22 > 0:
            measure_low = (size_10*measure_10 + size_21*measure_21 + size_00*measure_00 + size_02*measure_02 + size_11*measure_11 + size_20*measure_20 + size_22*measure_22)/(size_10 + size_21 + size_00 + size_02 + size_11 + size_20 + size_22)
        else:
            measure_low = np.mean([measure_10, measure_21, measure_00, measure_02, measure_11, measure_20, measure_22])  # all sizes 0: revert to basic measure

        measure = measure_high - measure_low

        if size_10 + size_21 > 0:
            measure_high_transpose = (size_10*measure_10 + size_21*measure_21)/(size_10 + size_21)
        else:
            measure_high_transpose = np.mean([measure_10, measure_21])  # both sizes 0: revert to basic measure
        
        if size_01 + size_12 + size_00 + size_20 + size_11 + size_02 + size_22 > 0:
            measure_low_transpose = (size_01*measure_01 + size_12*measure_12 + size_00*measure_00 + size_20*measure_20 + size_11*measure_11 + size_02*measure_02 + size_22*measure_22)/(size_01 + size_12 + size_00 + size_20 + size_11 + size_02 + size_22)
        else:
            measure_low_transpose = np.mean([measure_01, measure_12, measure_00, measure_20, measure_11, measure_02, measure_22])  # all sizes 0: revert to basic measure
        
        measure_transpose = measure_high_transpose - measure_low_transpose

    return measure, measure_transpose

def define_gargaml_scores_directed(results_df_measures, score_type="basic"):
    nodes = []
    gargaml = []
    transposed_gargaml = []
    max_gargaml = []

    for i,line in results_df_measures.iterrows():
        measure, measure_transpose = calculate_score_directed(line, score_type=score_type)

        nodes.append(line["node"])
        gargaml.append(measure)
        transposed_gargaml.append(measure_transpose)
        max_gargaml.append(max(measure, measure_transpose))

    dict_gargaml = {
        "node": nodes,
        "GARGAML": gargaml, 
        "GARGAML_transposed": transposed_gargaml,
        "GARGAML_max": max_gargaml
    }

    results_df = pd.DataFrame(dict_gargaml)
    results_df.set_index("node", inplace=True)
    return results_df

def calculate_score_undirected(line, score_type="basic"):
    measure_1 = line["measure_1"]
    measure_2 = line["measure_2"]
    measure_3 = line["measure_3"]
    
    if score_type == "basic":
        measure = measure_2 - (measure_1 + measure_3) / 2
    
    elif score_type == "weighted_average":
        size_1 = line["size_1"]
        size_2 = line["size_2"]
        size_3 = line["size_3"]
        total_size = size_1 + size_3
        if total_size > 0:
            measure = measure_2 - (size_1 * measure_1 + size_3 * measure_3) / total_size
        elif size_2 > 0:
            measure = measure_2  # size_1/size_3 both 0, so only measure_2 is relevant
        else:
            measure = -1  # no neighbours at all: far away from smurfing
    return measure

def define_gargaml_scores_undirected(results_df_measures, score_type="basic"):
    nodes = results_df_measures["node"].tolist()
    gargaml = [calculate_score_undirected(line, score_type=score_type) for _, line in results_df_measures.iterrows()]

    dict_gargaml = {
        "node": nodes,
        "GARGAML": gargaml
    }

    results_df = pd.DataFrame(dict_gargaml)
    results_df.set_index("node", inplace=True)
    return results_df

def define_gargaml_scores(results_df_measures, directed, score_type="basic"):
    if directed:
        return define_gargaml_scores_directed(results_df_measures, score_type=score_type)
    else:
        return define_gargaml_scores_undirected(results_df_measures, score_type=score_type)
    
def summarise_gargaml_scores(G_reduced, df_results, columns = ["GARGAML"]):
    G_degree_dict = dict(G_reduced.degree())
    nodes = list(df_results.index)

    gargaml_values = dict(zip(df_results.index, df_results["GARGAML"]))

    summaries_neighbourhood = dict()
    summaries_degree = dict()

    for node in nodes:
        summaries_neighbourhood[node] = summaries_neighbourhoors_node(node, G_reduced, gargaml_values)
        summaries_degree[node] = degree_neighbours_node(node, G_reduced, G_degree_dict)
    
    GARGAML_df = combine_GARG_AML(G_reduced, gargaml_values, summaries_neighbourhood, summaries_degree)

    return GARGAML_df[columns]