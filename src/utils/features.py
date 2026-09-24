"""
Feature column groups for the GARG-AML tree/boosting models.

The tree models are fed a mixture of signals, and the ablations isolate
where the lift comes from. This module makes the mixture explicit: four
column groups, and named configurations that select subsets of them. Every
script that trains a tree or a boosting model asks here for its columns
instead of carrying its own list.

The groups
----------
=====  ==========================================================
``a``  The aggregated GARG-AML score itself (Eq. 8 / Eq. 14).
``b``  Raw block densities *and* block sizes -- the per-node
       numbers the score is built from, before aggregation.
       Direction-dependent: 3 blocks undirected, 9 directed.
``c``  Own degree plus the min/mean/max/std of the neighbours'
       degrees, on the Louvain-reduced graph. Pure topology, no
       GARG-AML signal at all, and the same for both directions.
``d``  min/mean/max/std of the neighbours' GARG-AML scores.
=====  ==========================================================

The configurations
------------------
``full`` is the model the paper reports, and it is **(a) + (c) + (d)**: it
receives no raw block densities or sizes. ``all`` is the genuinely
all-four config, ``blocks`` the (b)-only ablation and ``topology`` the
(c)-only one.

Output schema
-------------
The config name is written to the ``features`` column of the tidy CSV (see
src/utils/evaluation.py) and, via :func:`config_suffix`, into the result
filenames, so these strings are part of the output schema rather than an
implementation detail; ``full`` carries no suffix.
:func:`feature_schema` emits the per-config appendix table.
"""

from __future__ import annotations

import pandas as pd

# Group (a): the aggregated score.
GROUP_SCORE = ["GARGAML"]

# Group (b): block densities + block sizes, straight out of
# results/<dataset>_GARGAML_<direction>.csv. The undirected analysis has
# 3 blocks, the directed one a 3x3 grid.
GROUP_BLOCKS_UNDIRECTED = [
    "measure_1", "measure_2", "measure_3",
    "size_1", "size_2", "size_3",
]

GROUP_BLOCKS_DIRECTED = (
    [f"measure_{i}{j}" for i in range(3) for j in range(3)]
    + [f"size_{i}{j}" for i in range(3) for j in range(3)]
)

# Group (c): the topology-only group, with no GARG-AML signal anywhere in
# it. These degrees are measured on the **Louvain-reduced** graph, not the
# raw transaction graph: the ablation asks what a tree can do with topology
# alone given the same pre-processing GARG-AML gets, so the comparison
# isolates the feature groups rather than confounding them with the
# edge-removal step.
GROUP_DEGREE = [
    "degree", "degree_min", "degree_max", "degree_mean", "degree_std",
]

# Group (d): neighbourhood aggregation of the score.
GROUP_SCORE_STATS = [
    "GARGAML_min", "GARGAML_max", "GARGAML_mean", "GARGAML_std",
]

GROUP_NAMES: dict[str, str] = {
    "a": "GARG-AML score",
    "b": "block densities + sizes",
    "c": "neighbour degree stats",
    "d": "neighbour score stats",
}

# Config name -> ordered group keys. Order matters: it is the order in
# which the columns reach the estimator.
FEATURE_CONFIGS: dict[str, tuple[str, ...]] = {
    "full":     ("a", "d", "c"),   # the model the paper reports
    "blocks":   ("b",),            # block-only ablation
    "topology": ("c",),            # topology-only ablation
    "all":      ("a", "b", "c", "d"),
}

# Groups whose columns do *not* change with the direction of the analysis.
# Groups (a), (b) and (d) derive from the direction-specific measures CSV.
# Group (c) does not: ``summarise_gargaml_scores`` is handed the *undirected*
# reduced graph in both cases, so the degree columns are identical between
# the directed and undirected runs.
DIRECTION_FREE_GROUPS = frozenset({"c"})

# Groups that need the Louvain-reduced graph. Group (b) comes straight out
# of the measures CSV, so a blocks-only run skips graph construction and
# Louvain entirely, which is where the run time goes.
NEIGHBOURHOOD_GROUPS = frozenset({"a", "c", "d"})


# ---------------------------------------------------------------------------
# Column lookup
# ---------------------------------------------------------------------------

def _check_config(config):
    """Raise a readable error for an unknown config name."""
    if config not in FEATURE_CONFIGS:
        raise KeyError(
            f"Unknown feature config {config!r}; expected one of "
            f"{sorted(FEATURE_CONFIGS)}."
        )


def is_direction_free(config):
    """True when ``config``'s feature matrix is identical for both directions.

    Such a config is run **once**, not once per direction: a second pass
    would refit the same matrix and write a second set of result files
    implying a directed/undirected distinction that does not exist. Derived
    from the group set rather than hard-coded.
    """
    _check_config(config)
    return set(FEATURE_CONFIGS[config]) <= DIRECTION_FREE_GROUPS


def needs_neighbourhood(config):
    """True when ``config`` requires the Louvain-reduced graph to be built."""
    _check_config(config)
    return bool(set(FEATURE_CONFIGS[config]) & NEIGHBOURHOOD_GROUPS)


def feature_groups(directed):
    """The four groups for one direction, keyed by group letter."""
    return {
        "a": list(GROUP_SCORE),
        "b": list(GROUP_BLOCKS_DIRECTED if directed else GROUP_BLOCKS_UNDIRECTED),
        "c": list(GROUP_DEGREE),
        "d": list(GROUP_SCORE_STATS),
    }


def feature_columns(config, directed):
    """Columns fed to the estimator for ``config``, in estimator order."""
    _check_config(config)
    groups = feature_groups(directed)
    return [column for key in FEATURE_CONFIGS[config] for column in groups[key]]


def all_feature_columns(configs, directed):
    """Union of the columns needed by ``configs``, without duplicates.

    One data preparation serves every config, so the graph is built and
    Louvain run once rather than once per ablation.
    """
    columns = []
    for config in configs:
        for column in feature_columns(config, directed):
            if column not in columns:
                columns.append(column)
    return columns


def config_suffix(config):
    """Result-file suffix for ``config``; empty for ``full``."""
    _check_config(config)
    return "" if config == "full" else "_" + config


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def _feature_kind(column):
    """Sub-type of a column, for the appendix table."""
    if column.startswith("measure_"):
        return "density"
    if column.startswith("size_"):
        return "size"
    if column.startswith("degree"):
        return "degree"
    if column == "GARGAML":
        return "score"
    return "score_stat"


def feature_schema(config, directed):
    """The exact feature matrix of ``config`` as a documented table.

    One row per column, in estimator order, so the row count is the
    dimension of the matrix. Written to
    ``results/<dataset>_<direction><suffix>_feature_schema.csv``.
    """
    _check_config(config)
    groups = feature_groups(directed)
    rows = []
    for key in FEATURE_CONFIGS[config]:
        for column in groups[key]:
            rows.append({
                "feature": column,
                "group": key,
                "group_name": GROUP_NAMES[key],
                "kind": _feature_kind(column),
            })
    return pd.DataFrame(rows, columns=["feature", "group", "group_name", "kind"])
