import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pandas as pd
import matplotlib.pyplot as plt

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels, combine_patterns_GARGAML
from src.methods.gargaml_scores import define_gargaml_scores
from src.utils.runtime import resolve_results_dir

plt.style.use('bmh')

RESULTS_DIR = resolve_results_dir()

def plot_patterns_GARGAML(results_df, laundering_df, dataset, directed, pattern_columns, name=None, score_type="basic"):
    laundering_df["GARGAML"] = combine_patterns_GARGAML(results_df, laundering_df)["GARGAML"]
    
    laundering_df["GARGAML_rounded"] = [round(x, 1) for x in laundering_df["GARGAML"]]


    dir_string = "directed" if directed else "undirected"
    laundering_df[laundering_df["GARGAML"]!=-2].groupby("GARGAML_rounded")[pattern_columns].mean().plot(alpha=0.7, figsize=(10, 7))
    if name is None:
        plt.title(dataset+" - "+dir_string)
        plt.savefig(RESULTS_DIR+"/"+dataset+"_GARGAML_"+dir_string+"_"+score_type+".pdf")
    else:
        plt.title(dataset+" - "+dir_string+" - "+name)
        plt.savefig(RESULTS_DIR+"/"+dataset+"_GARGAML_"+dir_string+"_"+name+"_"+score_type+".pdf")
    plt.close()

def main():
    dataset = "HI-Small"  
    directed = False
    score_type = "basic"

    str_directed = "directed" if directed else "undirected"
    results_df_measures = pd.read_csv(RESULTS_DIR+"/"+dataset+"_GARGAML_"+str_directed+".csv")

    results_df = define_gargaml_scores(results_df_measures, directed, score_type=score_type)

    transactions_df_extended, pattern_columns = define_ML_labels(
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    laundering_combined, laundering_from, laundering_to = summarise_ML_labels(transactions_df_extended,pattern_columns)

    plot_patterns_GARGAML(results_df, laundering_combined, dataset, directed, pattern_columns, name=None, score_type=score_type)
    plot_patterns_GARGAML(results_df, laundering_from, dataset, directed, pattern_columns, name='Sender', score_type=score_type)
    plot_patterns_GARGAML(results_df, laundering_to, dataset, directed, pattern_columns, name='Receiver', score_type=score_type)

if __name__ == "__main__":
    main()