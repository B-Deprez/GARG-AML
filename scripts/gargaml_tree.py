from sklearn import tree

import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import numpy as np
import pandas as pd

from src.data.pattern_construction import define_ML_labels, summarise_ML_labels, combine_patterns_GARGAML
from src.methods.gargaml_scores import define_gargaml_scores, summarise_gargaml_scores
from src.data.graph_construction import construct_IBM_graph
from src.utils.graph_processing import graph_community
from src.utils.evaluation import (
    SEED,
    cv_splits,
    evaluate_model,
    evaluate_scores,
    holdout_split,
    metric_records,
    model_scores,
    nan_metrics,
    write_folds,
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

# Task 7: set to 0 to reproduce the original published single 70/30 split
# (Tables 10-11) -- holdout_split, one fit per model, no fold column, legacy
# files unchanged. Set to 5 (or any >=2) to run 5-fold stratified CV instead:
# cv_splits, a pooled out-of-fold pass per model, and the fold-std companion
# matrices. Both modes write to the same legacy filenames, so switching this
# and rerunning overwrites the other mode's output -- copy results/ aside
# first if you want to keep both on disk at once.
N_FOLDS = 5

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

def _fit_and_record(X_train, y_train, X_test, y_test, models, context):
    """Fit every model on one train/test partition and return its records.

    Shared by both of run_config's modes (task 7): the single 70/30 split
    (N_FOLDS=0) calls this once per (cutoff, target); 5-fold CV calls it once
    per fold. Also returns each model's test-set scores/predictions, indexed
    by account -- needed to pool the CV folds into one out-of-fold pass,
    harmlessly unused (but cheap) when N_FOLDS=0.
    """
    n_test = len(y_test)
    n_pos = int(y_test.sum())

    records = metric_records(
        {"imbalance": y_train.mean()}, n_test=n_test, n_pos=n_pos, model="", **context
    )

    scores, preds = {}, {}
    for model_key, fit_model in models:
        try: # If too few labels, the model will not work. The cell is reported as NaN, with the reason
            clf = fit_model(X_train, y_train)
            metrics = evaluate_model(clf, X_test, y_test)
            scores[model_key] = pd.Series(model_scores(clf, X_test), index=X_test.index)
            preds[model_key] = pd.Series(clf.predict(X_test), index=X_test.index)
            status = "ok"
        except Exception as exc:
            print("    "+model_key+" skipped: "+repr(exc))
            metrics = nan_metrics()
            status = "skipped: "+str(exc)

        records += metric_records(
            metrics, status=status, n_test=n_test, n_pos=n_pos, model=model_key, **context
        )

    return records, scores, preds


def write_fold_partition(laundering_combined, dataset, cut_offs, targets, n_splits, seed=SEED):
    """Persist the N_FOLDS partition once (task 7).

    The split for a given (cutoff, target) depends only on the label vector,
    ``n_splits`` and the seed -- not on which feature config or direction
    trains on it -- so this is called once from main(), and run_config's own
    cv_splits calls reproduce the exact same partition deterministically
    without needing to share any state with this function. ``n_splits`` must
    be the same N_FOLDS run_config is using, or the persisted file would
    silently describe a different partition than the one actually trained on.
    """
    rows = []
    for cutoff in cut_offs:
        for target in targets:
            y = (laundering_combined[target] > cutoff).astype(int)
            try:
                splits = cv_splits(y, y, n_splits=n_splits, seed=seed)  # X unused beyond its length
            except ValueError as exc:
                print("    no folds for "+target+" @"+str(cutoff)+": "+repr(exc))
                continue
            for fold, _, test_idx in splits:
                for account in y.index[test_idx]:
                    rows.append(dict(account=account, cutoff=cutoff, target=target, fold=fold))

    return write_folds(rows, dataset)

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

            X_df = laundering_combined[gargaml_columns]
            y = (laundering_combined[target] > cutoff).astype(int)

            if N_FOLDS == 0: # original published setup: single 70/30 split
                try:
                    X_train, X_test, y_train, y_test = holdout_split(X_df, y, seed=seed)
                except Exception as exc: # Too few labels to even split: no models for this cell
                    print("    no split: "+repr(exc))
                    records += metric_records({"imbalance": 0.0}, status = "skipped: "+str(exc), model = "", **context)
                    for model_key, _ in models:
                        records += metric_records(nan_metrics(), status = "skipped: "+str(exc), model = model_key, **context)
                    continue

                fold_records, _, _ = _fit_and_record(X_train, y_train, X_test, y_test, models, context)
                records += fold_records
                continue

            # Task 7: 5-fold (or N_FOLDS-fold) stratified CV instead.
            try:
                splits = cv_splits(X_df, y, n_splits=N_FOLDS, seed=seed)
            except ValueError as exc: # Too few positives for N_FOLDS-fold CV
                print("    no CV split: "+repr(exc))
                records += metric_records({"imbalance": 0.0}, status = "skipped: "+str(exc), model = "", fold = np.nan, **context)
                for model_key, _ in models:
                    records += metric_records(nan_metrics(), status = "skipped: "+str(exc), model = model_key, fold = np.nan, **context)
                continue

            oof_scores = {model_key: pd.Series(np.nan, index=X_df.index) for model_key, _ in models}
            oof_preds = {model_key: pd.Series(np.nan, index=X_df.index) for model_key, _ in models}

            for fold, train_idx, test_idx in splits:
                X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                fold_records, scores, preds = _fit_and_record(
                    X_train, y_train, X_test, y_test, models, dict(context, fold=fold)
                )
                records += fold_records
                for model_key, _ in models:
                    if model_key in scores:
                        oof_scores[model_key].loc[X_test.index] = scores[model_key]
                        oof_preds[model_key].loc[X_test.index] = preds[model_key]

            # Pooled out-of-fold pass (fold=-1): one row per model, ranked over
            # the whole population -- the alert-queue headline numbers task 2's
            # ranking metrics actually want, not a 20% slice (see task 7's "K
            # population" note).
            pooled_context = dict(context, fold=-1)
            for model_key, _ in models:
                scores, preds = oof_scores[model_key], oof_preds[model_key]
                if scores.isna().any():
                    metrics = nan_metrics()
                    status = "skipped: incomplete out-of-fold coverage"
                else:
                    metrics = evaluate_scores(y.values, scores.values, y_pred=preds.values)
                    status = "ok"
                records += metric_records(
                    metrics, status = status, n_test = len(y), n_pos = int(y.sum()),
                    model = model_key, **pooled_context
                )

    return write_metrics(records, dataset, str_directed, suffix=suffix, write_std=(N_FOLDS >= 2))

def main():
    dataset = "HI-Small"
    score_type = "weighted_average"

    configs = list(FEATURE_CONFIGS) # full (published), blocks, topology, all

    # The reduced graph does not depend on the direction, so it is built
    # once for both passes instead of once per data_preparation call.
    G_reduced = reduced_graph(dataset) if any(needs_neighbourhood(c) for c in configs) else None

    if N_FOLDS >= 2:
        print("Transductive "+str(N_FOLDS)+"-fold CV (task 7): neighbour-score/degree "
              "summary features are computed on the full graph before folding; the "
              "temporal/inductive split stays deferred to discussion (R2-M6).")
    else:
        print("Single 70/30 holdout split (original published setup; task 7's CV is "
              "off -- set N_FOLDS >= 2 in this script to enable it).")

    done = set() # direction-free configs already run; see is_direction_free
    fold_partition_written = False

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

        # The account population and its order are identical regardless of
        # direction (see write_fold_partition), so the partition only needs
        # writing once, from whichever laundering_combined is built first.
        if N_FOLDS >= 2 and not fold_partition_written:
            path = write_fold_partition(laundering_combined, dataset, CUT_OFFS, TARGET_COLUMNS, N_FOLDS)
            print("  folds -> "+path)
            fold_partition_written = True

        for config in todo:
            run_config(laundering_combined, dataset, directed, config)
            if is_direction_free(config):
                done.add(config)

if __name__ == "__main__":
    main()