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
from src.data.bank_views import (bank_clients, parse_view, patterns_path,
                                 resolve_banks, trans_path)
from src.utils.graph_processing import parse_resolution, reduce_graph
from src.utils.evaluation import (
    CUT_OFFS,
    HEADLINE_CUTOFFS,
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
from src.utils.hyperparameters import write_hyperparameters
from src.utils.naming import gargaml_key, pretty_config
from src.utils.runtime import env_override, select_datasets, echo_config, resolve_results_dir

from sklearn import tree
from sklearn import ensemble

from pickle import dump

# Datasets to run, smallest first: a plain name is the full graph, "_bank<b>"
# / "_banktop<k>" a single-institution view, "_res<r>" a Louvain resolution and
# "_nolouvain" no reduction at all (see src/utils/graph_processing). Each entry
# overwrites its own result files in place, and one whose stage-1 measures are
# missing is skipped rather than failing, so entries can be commented out
# freely. LI-Large and the no-Louvain arms are the expensive runs.
DATASETS = ["HI-Small_bank012", "HI-Small_banktop50",
            "HI-Small_res1", "HI-Small_res5",
            "HI-Small",                        # default resolution, 10
            "HI-Small_res20", "HI-Small_res50",
            "LI-Large",
            "HI-Small_nolouvain", "LI-Large_nolouvain"]

# The pattern targets swept for every dataset and feature config. The matching
# label cut-offs come from src/utils/evaluation.py rather than being declared
# here: one canonical list, imported by every script that sweeps labels.
TARGET_COLUMNS = ['Is Laundering', 'FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK']

# Per-dataset reductions of that sweep, keyed by the *underlying* dataset so a
# bank view inherits its base's settings. LI-Large runs the headline cut-offs
# only, matching what scripts/graphsage_baseline.py does there so the two
# models stay comparable cell for cell.
DATASET_SETTINGS = {
    "LI-Large": dict(cut_offs=HEADLINE_CUTOFFS),
}


def dataset_settings(dataset):
    """The (cut_offs, targets) sweep for ``dataset``; see DATASET_SETTINGS.

    Falls back to the full default sweep, and resolves a bank view
    ("HI-Small_bank012") to its base dataset's entry.
    """
    overrides = DATASET_SETTINGS.get(parse_view(dataset)[0], {})
    return (overrides.get("cut_offs", CUT_OFFS),
            overrides.get("targets", TARGET_COLUMNS))

# 0 = a single 70/30 stratified holdout: holdout_split, one fit per model, no
# fold column. >=2 = that many stratified folds: cv_splits, a pooled
# out-of-fold pass per model, and the fold-std companion matrices. Both modes
# write the same filenames, so switching this and rerunning overwrites the
# other mode's output -- copy results/ aside to keep both on disk at once.
N_FOLDS = 5

# Slurm overrides; the constants above remain the documented defaults.
DATASETS = select_datasets(DATASETS)
N_FOLDS = env_override("n_folds", N_FOLDS, int)
RESULTS_DIR = resolve_results_dir()

# Defaults for gargaml_tree/gargaml_boosting's own save_path, under RESULTS_DIR.
DEFAULT_TREE_SAVE_PATH = RESULTS_DIR+"/model_tree.pkl"
DEFAULT_BOOST_SAVE_PATH = RESULTS_DIR+"/model_boosting.pkl"

def gargaml_tree(X, y, save = False, save_path = None):
    save_path = save_path or DEFAULT_TREE_SAVE_PATH
    # random_state is required, not cosmetic: sklearn permutes features at every
    # split, so when two splits tie on the criterion the winner is drawn at
    # random and an unseeded tree is not reproducible run to run.
    clf = tree.DecisionTreeClassifier(min_samples_leaf=10, random_state=1997)
    clf = clf.fit(X, y)

    if save:
        with open(save_path, "wb") as f:
            dump(clf, f, protocol=5)

    return clf

def gargaml_boosting(X, y, save = False, save_path = None):
    save_path = save_path or DEFAULT_BOOST_SAVE_PATH
    clf = ensemble.GradientBoostingClassifier(min_samples_leaf=10, random_state=1997)
    clf = clf.fit(X, y)

    if save:
        with open(save_path, "wb") as f:
            dump(clf, f, protocol=5)

    return clf

def measures_path(dataset, directed):
    """Where stage 1 wrote ``dataset``'s block measures for this direction."""
    return RESULTS_DIR+"/"+dataset+"_GARGAML_"+("directed" if directed else "undirected")+".csv"

def available_directions(dataset):
    """Directions of ``dataset`` whose stage-1 measures are actually on disk.

    This script is stage 2 of a decoupled pipeline: it reads what
    gargaml_directed.py / gargaml_undirected.py wrote. A dataset listed in
    DATASETS whose measures were never computed -- a bank view is the usual
    case, since the measure scripts have to be run on the view name first --
    is reported and skipped, so the loop carries on with the datasets that
    are ready.
    """
    return [d for d in [False, True] if os.path.exists(measures_path(dataset, d))]

def reduced_graph(dataset):
    """The Louvain-reduced graph the neighbourhood summaries are computed on.

    Built from the undirected transaction graph whichever measures CSV is
    being scored, so it is identical across the directed and undirected
    passes -- build it once in the caller and hand it to both.
    """
    base, banks = parse_view(dataset)  # None for the full graph
    G = construct_IBM_graph(path = trans_path(dataset), directed = False, banks = banks)
    # Same Louvain setting stage 1 used, read back out of the dataset name, so
    # the neighbourhood features match the measures they are joined to. dataset
    # is not passed on: stage 1 already logged the severance and this would
    # duplicate the row.
    return reduce_graph(G, parse_resolution(dataset)[1], results_dir=RESULTS_DIR)

def data_preparation(dataset, feature_cols, directed, score_type, G_reduced = None):
    """Build one feature table holding every column in ``feature_cols``.

    ``feature_cols`` is the *union* over the feature configs that will be
    run (see src/utils/features.py), so the graph is built and Louvain is
    run once for all of them.

    Pass ``G_reduced`` to reuse a graph already built by :func:`reduced_graph`
    (it does not depend on ``directed``); leave it ``None`` to build on
    demand, which is what a single-config caller wants.
    """
    str_directed = "directed" if directed else "undirected"
    base, banks = parse_view(dataset) #None for the full graph
    # Expand a group spec ("top50") into its member banks before anything
    # filters on it: bank_clients matches against the bank column and would
    # otherwise select nobody. resolve_banks caches per (path, spec).
    banks = resolve_banks(banks, trans_path(dataset))

    results_df_measures = pd.read_csv(measures_path(dataset, directed)) #measures

    results_df = define_gargaml_scores(results_df_measures, directed, score_type=score_type) #summary scores

    # Group (b) is already in the measures frame, so carrying the block
    # densities and sizes costs one join rather than a second read. Group
    # membership decides which columns come from here, not what the CSV
    # happens to be called.
    block_columns = [c for c in feature_cols if c in feature_groups(directed)["b"]]
    if block_columns:
        results_df = results_df.join(results_df_measures.set_index("node")[block_columns])

    transactions_df_extended, pattern_columns = define_ML_labels( #patterns
        path_trans = trans_path(dataset),
        path_patterns = patterns_path(dataset),
        banks = banks
    )

    summary_columns = [c for c in feature_cols if c not in block_columns] #groups (a), (c), (d)
    if summary_columns: #a block-only run needs no neighbourhood summaries, so skip Louvain
        if G_reduced is None:
            G_reduced = reduced_graph(dataset) #undirected: see that function

        summary_gargaml = summarise_gargaml_scores(G_reduced, results_df, columns = summary_columns)
        for column in summary_columns:
            results_df[column] = summary_gargaml[column]

    laundering_combined, _, _ = summarise_ML_labels(transactions_df_extended,pattern_columns)

    if banks is not None:
        # A bank alerts on its own customers, so they are the evaluated
        # population -- not the external counterparties, which stay in the
        # graph as neighbours but are not scored. This also keeps the labels
        # honest: a bank sees every transaction of its own clients, so their
        # propensities here equal the full-data ones and only the features
        # differ. Restricting before combine_patterns_GARGAML also spares it
        # the dropped accounts.
        clients = bank_clients(transactions_df_extended, banks)
        laundering_combined = laundering_combined[laundering_combined.index.isin(clients)]
        print("  bank view: "+str(len(laundering_combined))+" client accounts evaluated")

    combined_patterns_GARGAML = combine_patterns_GARGAML(results_df, laundering_combined, columns = feature_cols)

    for column in feature_cols:
        laundering_combined[column] = combined_patterns_GARGAML[column]

    return laundering_combined

def _fit_and_record(X_train, y_train, X_test, y_test, models, context):
    """Fit every model on one train/test partition and return its records.

    Shared by both of run_config's modes: the single 70/30 split (N_FOLDS=0)
    calls this once per (cutoff, target); cross-validation calls it once per
    fold. Also returns each model's test-set scores/predictions, indexed by
    account, which is what pools the CV folds into one out-of-fold pass.
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
    """Persist the N_FOLDS partition once.

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

    return write_folds(rows, dataset, results_dir=RESULTS_DIR)

def run_config(laundering_combined, dataset, directed, config, seed=SEED,
               cut_offs=None, targets=None):
    """Train and evaluate both models on one feature config.

    The feature config is the only thing that varies: the split, the
    cut-off/target sweep and the estimators are identical, which is what
    makes the ablation readable. Results go to the ``config``-specific
    suffix, and ``full`` writes the unsuffixed filenames.

    ``cut_offs`` / ``targets`` default to whatever
    :func:`dataset_settings` resolves for ``dataset``, so a caller that
    does not care (scripts/gargaml_tree_blocks.py) picks up a per-dataset
    reduction automatically instead of silently running the full grid.
    """
    str_directed = "directed" if directed else "undirected"
    suffix = config_suffix(config)
    gargaml_columns = feature_columns(config, directed)

    default_cut_offs, default_targets = dataset_settings(dataset)
    cut_offs = default_cut_offs if cut_offs is None else cut_offs
    targets = default_targets if targets is None else targets

    print("\n=== "+dataset+" ("+str_directed+"), features: "+config+" ===")
    print("  models: "+", ".join(pretty_config(gargaml_key(v, directed), config) for v in ["tree", "boost"]))
    print("  features ("+str(len(gargaml_columns))+"): "+str(gargaml_columns))
    print("  sweep: "+str(len(cut_offs))+" cut-offs x "+str(len(targets))+" targets"
          +(" (reduced; see DATASET_SETTINGS)"
            if (cut_offs, targets) != (CUT_OFFS, TARGET_COLUMNS) else ""))

    # Persist the exact features the models see, for the appendix.
    schema_path = RESULTS_DIR+"/"+dataset+"_"+str_directed+suffix+"_feature_schema.csv"
    feature_schema(config, directed).to_csv(schema_path, index=False)
    print("  feature schema -> "+schema_path)

    models = [
        (gargaml_key("tree", directed), gargaml_tree),
        (gargaml_key("boost", directed), gargaml_boosting),
    ]

    records = []

    # Iterate the *full* default grid and skip what this dataset's sweep
    # leaves out, rather than iterating the reduced sweep directly. Two
    # reasons, both about the metric matrices:
    #   * write_metric_matrices takes its row/column order from the values
    #     actually present, and VisualisationResults.ipynb indexes those
    #     files with df.loc[cut_off][pattern] over the full cut-off list, so
    #     a reduced sweep is a KeyError there, not a smaller table.
    #   * a cell that was never attempted is a gap like any other, and gaps
    #     are reported rather than dropped. It goes through the same
    #     nan_metrics() path as a too-few-positives cell, with a status that
    #     says which of the two it was.
    for cutoff in CUT_OFFS:
        for target in TARGET_COLUMNS:
            context = dict(
                dataset = dataset,
                direction = str_directed,
                features = config, # which column groups the models see; see
                                    # src/utils/features.py for what each config
                                    # holds
                cutoff = cutoff,
                target = target,
                seed = seed,
            )

            if cutoff not in cut_offs or target not in targets:
                reason = "skipped: not in this dataset's sweep (DATASET_SETTINGS)"
                fold_ctx = {} if N_FOLDS == 0 else dict(fold = np.nan)
                records += metric_records({"imbalance": np.nan}, status = reason,
                                          model = "", **fold_ctx, **context)
                for model_key, _ in models:
                    records += metric_records(nan_metrics(), status = reason,
                                              model = model_key, **fold_ctx, **context)
                continue

            print(cutoff, target)

            X_df = laundering_combined[gargaml_columns]
            y = (laundering_combined[target] > cutoff).astype(int)

            if N_FOLDS == 0: # single 70/30 stratified holdout
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

            # N_FOLDS-fold stratified cross-validation instead.
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
            # the whole population rather than a single fold's slice, which is
            # the population the alert-queue ranking metrics are defined on.
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

    return write_metrics(records, dataset, str_directed, suffix=suffix,
                         results_dir=RESULTS_DIR, write_std=(N_FOLDS >= 2))

def main():
    for dataset in DATASETS:
        run_dataset(dataset)

def run_dataset(dataset):
    score_type = "weighted_average"

    configs = list(FEATURE_CONFIGS) # full, blocks, topology, all

    # Resolved once and threaded everywhere, so the persisted fold partition
    # covers exactly the (cutoff, target) cells that will be trained on. A
    # partition written over the full grid while the models run a reduced one
    # would describe folds nothing reads, and GraphSAGE reads this file.
    cut_offs, targets = dataset_settings(dataset)

    # Stage 1 must have run on this dataset name -- including on a bank view's
    # name, which is a dataset of its own. Checked before the graph is built,
    # which is the expensive part.
    directions = available_directions(dataset)
    if not directions:
        print("\n### "+dataset+" -- SKIPPED: no block measures on disk ("
              +measures_path(dataset, False)+"). Run gargaml_undirected.py / "
              "gargaml_directed.py on this dataset name first. ###")
        return
    for directed in [False, True]:
        if directed not in directions:
            print("\n### "+dataset+" ("+("directed" if directed else "undirected")
                  +") -- SKIPPED: "+measures_path(dataset, directed)+" not found ###")

    # Record what every estimator is configured with, and that none of it is
    # searched or selected on the test split. Written up front rather than at
    # the end, so an interrupted run still documents the configuration its
    # partial results came from.
    print("hyperparameters -> "+write_hyperparameters(dataset, results_dir=RESULTS_DIR))

    # The reduced graph does not depend on the direction, so it is built
    # once for both passes instead of once per data_preparation call.
    G_reduced = reduced_graph(dataset) if any(needs_neighbourhood(c) for c in configs) else None

    if (cut_offs, targets) != (CUT_OFFS, TARGET_COLUMNS):
        print("Reduced sweep for "+dataset+" (DATASET_SETTINGS): "
              +str(len(cut_offs))+" of "+str(len(CUT_OFFS))+" cut-offs, "
              +str(len(targets))+" of "+str(len(TARGET_COLUMNS))+" targets. The "
              "omitted cells are written as NaN with a 'not in this dataset's "
              "sweep' status, not dropped.")

    if N_FOLDS >= 2:
        print("Transductive "+str(N_FOLDS)+"-fold CV: neighbour-score/degree "
              "summary features are computed on the full graph before folding, so "
              "the evaluation is transductive rather than inductive.")
    else:
        print("Single 70/30 holdout split -- set N_FOLDS >= 2 in this script for "
              "cross-validation instead.")

    done = set() # direction-free configs already run; see is_direction_free
    fold_partition_written = False

    for directed in directions:
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
            path = write_fold_partition(laundering_combined, dataset, cut_offs, targets, N_FOLDS)
            print("  folds -> "+path)
            fold_partition_written = True

        for config in todo:
            run_config(laundering_combined, dataset, directed, config,
                       cut_offs=cut_offs, targets=targets)
            if is_direction_free(config):
                done.add(config)

if __name__ == "__main__":
    echo_config(__file__, datasets=DATASETS, n_folds=N_FOLDS,
                cut_offs=CUT_OFFS, targets=TARGET_COLUMNS, results_dir=RESULTS_DIR)
    main()