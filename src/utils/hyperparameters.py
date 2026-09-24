"""
Hyperparameter provenance for every fitted model.

A registry rather than a tuning log, because **no hyperparameter search is
run**: every value is either a library default inherited untouched or a
single value fixed a priori, and no selection procedure consults the test
split. This module states that in the form the appendix table takes,
instead of leaving a reader to reconstruct it from constructor calls
scattered across five scripts.

Three design points:

1. **Defaults are read at runtime, not transcribed.** Most of the
   configuration is whatever scikit-learn's defaults are: the paper's
   "100 trees, max_depth 3, learning_rate 0.1" is
   ``GradientBoostingClassifier``'s default triple, not a chosen setting.
   A hardcoded copy would go stale on the next scikit-learn upgrade and
   the appendix would then describe a run that never happened, so
   :func:`hyperparameter_schema` instantiates each estimator and reads
   ``get_params()``. The same holds for the Louvain resolution and the
   split sizes, read from the signatures of
   :func:`~src.utils.graph_processing.graph_community` and
   :func:`~src.utils.evaluation.holdout_split`. The ``source`` column
   marks which values this repository sets (``explicit``) and which come
   from the library (``library_default``).

2. **Seeds are not hyperparameters.** ``random_state=1997`` appears in
   every constructor but controls reproducibility, not model capacity, so
   it is reported with ``role="seed"`` and can be filtered out of a
   hyperparameter table without special-casing the name. The same applies
   to bookkeeping parameters such as ``verbose``.

3. **The two directions share one configuration.** ``gargaml_tree_u`` and
   ``gargaml_tree_d`` are the same estimator on different feature
   matrices, so they get identical rows rather than a shared row -- the
   ``component`` column then joins directly onto the ``model`` column of
   the tidy results CSV (src/utils/evaluation.py).

Call sites covered: ``scripts/gargaml_tree.py``, ``gargaml_tree_blocks.py``,
``gargaml_tree_synthetic{,_3,_5}.py`` (all five construct the estimators
with the identical arguments registered here), ``gargaml_IF.py`` and
``scripts/graphsage_baseline.py``.

Runnable standalone::

    python -m src.utils.hyperparameters

so the appendix table can be regenerated without re-running a fit.
"""

from __future__ import annotations

import inspect

import pandas as pd

from sklearn import ensemble, tree

from src.utils.evaluation import CV_FOLDS, SEED, holdout_split
from src.utils.graph_processing import graph_community
from src.utils.naming import pretty

# The three provenance answers. They are identical for every row in this
# registry, and are repeated per row so the CSV is self-contained.
SEARCH_SPACE = "not searched"
SELECTION_CRITERION = "none -- no model selection was performed"
TUNED_ON_TEST = "no"

# Parameters that exist in a constructor but do not describe the model:
# reproducibility seeds and bookkeeping. Reported, but flagged so an
# appendix table can drop them.
SEED_PARAMS = {"random_state", "seed"}
BOOKKEEPING_PARAMS = {"verbose", "verbose_interval", "n_jobs", "warm_start"}

# What each script passes explicitly. Everything else in the estimator is
# a library default, read at runtime rather than copied out here.
TREE_PARAMS = {"min_samples_leaf": 10, "random_state": SEED}
BOOST_PARAMS = {"min_samples_leaf": 10, "random_state": SEED}
IF_PARAMS = {"random_state": SEED}

_SKLEARN_MODELS = [
    ("gargaml_tree_u", tree.DecisionTreeClassifier, TREE_PARAMS),
    ("gargaml_tree_d", tree.DecisionTreeClassifier, TREE_PARAMS),
    ("gargaml_boost_u", ensemble.GradientBoostingClassifier, BOOST_PARAMS),
    ("gargaml_boost_d", ensemble.GradientBoostingClassifier, BOOST_PARAMS),
    ("gargaml_if_d", ensemble.IsolationForest, IF_PARAMS),
]

# GraphSAGE. Declared rather than read from the estimator, because
# importing src.methods.graphsage pulls in torch and torch-geometric and
# this module stays importable without them.
# :func:`check_graphsage_declaration` verifies the declaration against
# ``train_fold``'s actual signature whenever torch *is* available, so the
# two cannot drift apart silently.
GRAPHSAGE_PARAMS = {
    "hidden_channels": 64,
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 1024,
    "num_neighbors": (25, 10),
    # ``epochs`` is an upper bound, not a chosen training length: early
    # stopping on validation AUC-PR decides the real number, and the
    # epochs actually run are reported per fold in the results CSV.
    "epochs": 50,
    "patience": 5,
    "val_fraction": 0.1,
    "seed": SEED,
}

# Structural choices, not tunable arguments of train_fold.
GRAPHSAGE_ARCHITECTURE = {
    "layers": 2,
    "conv": "SAGEConv",
    "optimizer": "Adam",
    "loss": "BCEWithLogitsLoss(pos_weight=n_neg/n_pos)",
}

SCHEMA_COLUMNS = [
    "component", "component_name", "kind", "parameter", "value",
    "source", "role", "search_space", "selection_criterion",
    "tuned_on_test", "note",
]


def _role(parameter):
    """Whether ``parameter`` describes the model or merely its bookkeeping."""
    if parameter in SEED_PARAMS:
        return "seed"
    if parameter in BOOKKEEPING_PARAMS:
        return "bookkeeping"
    return "hyperparameter"


def _row(component, component_name, kind, parameter, value, source,
         role=None, note=""):
    return {
        "component": component,
        "component_name": component_name,
        "kind": kind,
        "parameter": parameter,
        "value": str(value),
        "source": source,
        "role": role or _role(parameter),
        "search_space": SEARCH_SPACE,
        "selection_criterion": SELECTION_CRITERION,
        "tuned_on_test": TUNED_ON_TEST,
        "note": note,
    }


def _default(func, parameter):
    """The default value of ``parameter`` in ``func``'s signature."""
    return inspect.signature(func).parameters[parameter].default


def _sklearn_rows(key, estimator_cls, explicit):
    """One row per constructor parameter of a fitted scikit-learn model.

    The full parameter set comes from ``get_params()`` on an instance
    built exactly as the scripts build it, so the table describes the
    model that actually ran rather than the arguments someone remembered
    to pass.
    """
    clf = estimator_cls(**explicit)
    name = pretty(key)
    rows = []
    for parameter, value in sorted(clf.get_params().items()):
        source = "explicit" if parameter in explicit else "library_default"
        rows.append(_row(key, name, "model", parameter, value, source))
    return rows


def _graphsage_rows():
    key = "graphsage_u"
    name = pretty(key)
    rows = [
        _row(key, name, "model", parameter, value, "explicit")
        for parameter, value in sorted(GRAPHSAGE_PARAMS.items())
    ]
    rows += [
        _row(key, name, "model", parameter, value, "explicit", role="architecture")
        for parameter, value in sorted(GRAPHSAGE_ARCHITECTURE.items())
    ]
    return rows


def _preprocessing_rows():
    """Louvain, reported alongside the model hyperparameters.

    The resolution is read from ``graph_community``'s signature rather than
    restated, so this row cannot claim a value the pipeline is not using.
    """
    note = ("chosen a priori for smaller communities; swept by the "
            "_res<r> dataset arms")
    return [
        _row("louvain", "Louvain community detection", "preprocessing",
             "resolution", _default(graph_community, "resolution"),
             "explicit", note=note),
        _row("louvain", "Louvain community detection", "preprocessing",
             "seed", SEED, "explicit"),
        _row("louvain", "Louvain community detection", "preprocessing",
             "edge_filter", "intra-community edges only", "explicit",
             role="architecture",
             note="Algorithm 1; the % of edges severed is quantified in "
                  "notebooks/LouvainEdgeSeverance.ipynb"),
    ]


def _evaluation_rows():
    """The split protocol, for the "did tuning see the test set" question.

    Not hyperparameters, but the table is where a reader looks for them,
    and both values are read from the code that performs the split.
    """
    return [
        _row("split", "Evaluation protocol", "evaluation", "test_size",
             _default(holdout_split, "test_size"), "explicit",
             role="protocol", note="single stratified holdout (synthetic runs)"),
        _row("split", "Evaluation protocol", "evaluation", "n_splits",
             CV_FOLDS, "explicit", role="protocol",
             note="stratified CV on the IBM data"),
        _row("split", "Evaluation protocol", "evaluation", "seed", SEED,
             "explicit"),
    ]


def hyperparameter_schema():
    """Every fitted model's configuration, with its provenance.

    One row per parameter. ``source`` separates the values this repository
    sets from the library defaults it inherits; ``role`` separates real
    hyperparameters from seeds, bookkeeping and structural choices. An
    appendix table is this frame filtered to
    ``role == "hyperparameter"``.
    """
    rows = []
    for key, estimator_cls, explicit in _SKLEARN_MODELS:
        rows += _sklearn_rows(key, estimator_cls, explicit)
    rows += _graphsage_rows()
    rows += _preprocessing_rows()
    rows += _evaluation_rows()
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


def check_graphsage_declaration():
    """Verify GRAPHSAGE_PARAMS against ``train_fold``'s real defaults.

    Returns a list of human-readable mismatches, or an empty list when the
    declaration is faithful -- including when torch is not installed, in
    which case there is nothing to check against and the declaration is
    taken at face value. GRAPHSAGE_PARAMS is the one part of the registry
    that is copied rather than read, and a stale copy would misreport a run.
    """
    try:
        from src.methods.graphsage import train_fold
    except ImportError:
        return []

    signature = inspect.signature(train_fold).parameters
    mismatches = []
    for parameter, declared in GRAPHSAGE_PARAMS.items():
        if parameter not in signature:
            mismatches.append(parameter+": declared here but not a train_fold argument")
            continue
        actual = signature[parameter].default
        # num_neighbors is declared as a tuple but train_fold may carry a
        # list (or vice versa); compare by content, not by container type.
        if isinstance(declared, (tuple, list)) and isinstance(actual, (tuple, list)):
            differs = tuple(declared) != tuple(actual)
        else:
            differs = declared != actual
        if differs:
            mismatches.append(parameter+": declared "+repr(declared)+", train_fold default "+repr(actual))
    return mismatches


def write_hyperparameters(dataset, results_dir="results"):
    """Write the registry to ``results/<dataset>_hyperparameters.csv``.

    Called once per run beside the per-config feature schema, so every
    result directory carries the configuration that produced it.
    """
    path = results_dir+"/"+dataset+"_hyperparameters.csv"
    hyperparameter_schema().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    schema = hyperparameter_schema()
    mismatches = check_graphsage_declaration()
    for mismatch in mismatches:
        print("WARNING GraphSAGE declaration drift -- "+mismatch)

    chosen = schema[(schema["source"] == "explicit") & (schema["role"] == "hyperparameter")]
    print(chosen[["component", "parameter", "value"]].to_string(index=False))
    print("\n"+str(len(schema))+" parameters registered, "+str(len(chosen))+" set explicitly.")
    print("Search space: "+SEARCH_SPACE+". Selection: "+SELECTION_CRITERION+".")
    print("Tuning that used the test split: "+TUNED_ON_TEST+".")
