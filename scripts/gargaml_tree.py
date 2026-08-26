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
from src.utils.features import (
    FEATURE_CONFIGS,
    all_feature_columns,
    config_suffix,
    feature_columns,
    feature_groups,
    feature_schema,
    is_direction_free,
    needs_neighbourhood,
)
from src.utils.naming import gargaml_key, pretty_config

from sklearn import tree
from sklearn import ensemble

from pickle import dump

# The label cut-offs and the pattern targets are swept identically by every
# feature config, so they live here rather than being re-declared per config.
CUT_OFFS = [0.1, 0.2, 0.3, 0.5, 0.9]
TARGET_COLUMNS = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

def gargaml_tree(X, y, save = False, save_path = "results/model_tree.pkl"):
    # random_state is required, not cosmetic: sklearn permutes features at every
    # split, so when two splits tie on the criterion the winner is drawn at
    # random and an unseeded tree is not reproducible run to run. This script
    # was the only one missing it (the synthetic scripts always passed 1997),
    # which is why re-running it moved cells that no code change should touch.
    # Task 3's ablation and task 7's fold variance both compare tree runs, so
    # unseeded tie-breaking would show up as signal in both.
    clf = tree.DecisionTreeClassifier(min_samples_leaf=10, random_state=1997)
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

def reduced_graph(dataset):
    """The Louvain-reduced graph the neighbourhood summaries are computed on.

    Built from the **undirected** transaction graph whichever measures CSV
    is being scored, so it is identical across the directed and undirected
    passes -- build it once in the caller and hand it to both. Graph
    construction plus Louvain is where this script's run time goes; the
    model fits are cheap next to it.
    """
    G = construct_IBM_graph(path = "data/"+dataset+"_Trans.csv", directed = False)
    return graph_community(G)

def data_preparation(dataset, feature_cols, directed, score_type, G_reduced = None):
    """Build one feature table holding every column in ``feature_cols``.

    ``feature_cols`` is the *union* over the feature configs that will be
    run (see src/utils/features.py), so the graph is built and Louvain is
    run once for all of them -- that is where the run time of this script
    goes, not in the model fits.

    Pass ``G_reduced`` to reuse a graph already built by :func:`reduced_graph`
    (it does not depend on ``directed``); leave it ``None`` to build on
    demand, which is what a single-config caller wants.
    """
    str_directed = "directed" if directed else "undirected"

    results_df_measures = pd.read_csv("results/"+dataset+"_GARGAML_"+str_directed+".csv") #measures

    results_df = define_gargaml_scores(results_df_measures, directed, score_type=score_type) #summary scores

    # Group (b) is already in the measures frame, so carrying the block
    # densities and sizes costs one join rather than a second read. Group
    # membership decides which columns come from here, not what the CSV
    # happens to be called: a new measure column must not silently bypass
    # the neighbourhood summary below.
    block_columns = [c for c in feature_cols if c in feature_groups(directed)["b"]]
    if block_columns:
        results_df = results_df.join(results_df_measures.set_index("node")[block_columns])

    transactions_df_extended, pattern_columns = define_ML_labels( #patterns
        path_trans = "data/"+dataset+"_Trans.csv",
        path_patterns = "data/"+dataset+"_Patterns.txt"
    )

    summary_columns = [c for c in feature_cols if c not in block_columns] #groups (a), (c), (d)
    if summary_columns: #a block-only run needs no neighbourhood summaries, so skip Louvain
        if G_reduced is None:
            G_reduced = reduced_graph(dataset) #undirected: see that function

        summary_gargaml = summarise_gargaml_scores(G_reduced, results_df, columns = summary_columns)
        for column in summary_columns:
            results_df[column] = summary_gargaml[column]

    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)

    combined_patterns_GARGAML = combine_patterns_GARGAML(results_df, laundering_combined, columns = feature_cols)

    for column in feature_cols:
        laundering_combined[column] = combined_patterns_GARGAML[column]

    return laundering_combined

def data_split(laundering_combined, gargaml_columns, target, cutoff, seed=SEED):
    X_df = laundering_combined[gargaml_columns]
    rel_labels = laundering_combined[target]
    y = (rel_labels>cutoff)*1

    X_train, X_test, y_train, y_test = holdout_split(X_df, y, seed=seed)

    return X_train, X_test, y_train, y_test

def run_config(laundering_combined, dataset, directed, config, seed=SEED):
    """Train and evaluate both models on one feature config (task 3).

    The feature config is the only thing that varies: the split, the
    cut-off/target sweep and the estimators are identical, which is what
    makes the ablation readable. Results go to the ``config``-specific
    suffix, so the published ``full`` files keep their historical names.
    """
    str_directed = "directed" if directed else "undirected"
    suffix = config_suffix(config)
    gargaml_columns = feature_columns(config, directed)

    print("\n=== "+dataset+" ("+str_directed+"), features: "+config+" ===")
    print("  models: "+", ".join(pretty_config(gargaml_key(v, directed), config) for v in ["tree", "boost"]))
    print("  features ("+str(len(gargaml_columns))+"): "+str(gargaml_columns))

    # Persist the feature schema for the appendix (task 10).
    schema_path = "results/"+dataset+"_"+str_directed+suffix+"_feature_schema.csv"
    feature_schema(config, directed).to_csv(schema_path, index=False)
    print("  feature schema -> "+schema_path)

    models = [
        (gargaml_key("tree", directed), gargaml_tree),
        (gargaml_key("boost", directed), gargaml_boosting),
    ]

    records = []

    for cutoff in CUT_OFFS:
        for target in TARGET_COLUMNS:
            print(cutoff, target)

            context = dict(
                dataset = dataset,
                direction = str_directed,
                features = config, # which task-3 column groups the models see;
                                    # see src/utils/features.py for what each config holds
                                    # and why "full" is not literally all four groups
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

    return write_metrics(records, dataset, str_directed, suffix=suffix)

def main():
    dataset = "HI-Small"
    score_type = "weighted_average"

    configs = list(FEATURE_CONFIGS) # full (published), blocks, topology, all

    # The reduced graph does not depend on the direction, so it is built
    # once for both passes instead of once per data_preparation call.
    G_reduced = reduced_graph(dataset) if any(needs_neighbourhood(c) for c in configs) else None

    done = set() # direction-free configs already run; see is_direction_free

    for directed in [False, True]:
        str_directed = "directed" if directed else "undirected"

        # A direction-free config (topology) has an identical feature matrix
        # in both passes. Running it twice would refit the same matrix and
        # write a second set of result files implying a directed/undirected
        # distinction that does not exist, so it runs once and is announced
        # as skipped rather than silently dropped.
        todo = [c for c in configs if c not in done]
        for config in configs:
            if config in done:
                print("\n=== "+dataset+" ("+str_directed+"), features: "+config+
                      " -- SKIPPED: direction-free, already run ===")
        if not todo:
            continue

        print("\nData preparation ("+str_directed+")")
        laundering_combined = data_preparation(
            dataset, all_feature_columns(todo, directed), directed, score_type,
            G_reduced = G_reduced
            )

        for config in todo:
            run_config(laundering_combined, dataset, directed, config)
            if is_direction_free(config):
                done.add(config)

if __name__ == "__main__":
    main()