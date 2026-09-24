"""
Canonical model naming for GARG-AML results.

Single source of truth for the display names used in tables, figures and
legends across the paper, so every label comes from one mapping.

Two-tier rule:

* **Internal keys** (CSV columns, filename suffixes, dict keys, the raw
  ``method`` column in dataframes) are the lowercase snake-case identifiers
  in ``MODEL_DISPLAY_NAMES``, which is what the result files carry.
* **Display names** (legends, axis labels, table headers, CD-diagram
  labels, written text) come from :func:`pretty`.

A new model is one entry in ``MODEL_DISPLAY_NAMES`` plus its key appended to
``MODEL_ORDER``. The feature-group *ablations* are not new models -- they are
the same estimators on different feature groups, so they reuse these keys and
are labelled through :func:`pretty_config` instead.
"""

from __future__ import annotations

from typing import Iterable

# Canonical key -> display name mapping. Display names follow the
# Tables 10-11 convention from the paper: "GARG-AML <Direction>. <Variant>".
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "flowscope":       "FlowScope",
    "autoaudit":       "AutoAudit",
    "gargaml_u":       "GARG-AML Undirected",
    "gargaml_d":       "GARG-AML Directed",
    "gargaml_tree_u":  "GARG-AML Undir. Tree",
    "gargaml_boost_u": "GARG-AML Undir. Boost",
    "gargaml_tree_d":  "GARG-AML Dir. Tree",
    "gargaml_boost_d": "GARG-AML Dir. Boost",
    # Isolation-forest baseline (scripts/gargaml_IF.py): directed only, since it
    # reads the 9 directed block-density measures unconditionally.
    "gargaml_if_d":    "GARG-AML Isolation Forest",
    # GraphSAGE baseline (scripts/graphsage_baseline.py): undirected, run against
    # the undirected GARG-AML score, and never fed GARG-AML scores or block
    # measures.
    "graphsage_u":     "GraphSAGE",
}

# The feature configs (src/utils/features.py) are ablations of the *same*
# estimators, so they reuse the model keys above and are distinguished in
# tables by a label suffix rather than by new keys.
FEATURE_CONFIG_LABELS: dict[str, str] = {
    "full":   "",                  # the model the paper reports
    "blocks": " (blocks only)",
    "all":    " (all features)",
}

# ``topology`` is the exception and gets its own base name. That config
# contains **no GARG-AML signal at all** -- neighbour-degree statistics
# only -- so labelling it "GARG-AML ..." would misrepresent the ablation,
# whose point is what the tree achieves *without* GARG-AML. It is also
# direction-free (degrees come from the undirected reduced graph either
# way), so the Undir./Dir. distinction is dropped from the label.
TOPOLOGY_DISPLAY_NAMES: dict[str, str] = {
    "gargaml_tree_u":  "Degree-only Tree",
    "gargaml_tree_d":  "Degree-only Tree",
    "gargaml_boost_u": "Degree-only Boost",
    "gargaml_boost_d": "Degree-only Boost",
}

# Feature configs that belong to one model rather than to the ablation
# family. GraphSAGE's two configs are not ablations of a GARG-AML feature
# matrix -- they are the strict-parity and the deliberately generous input
# sets the baseline is reported under -- so they get their own labels instead
# of falling through to the "(config)" default.
MODEL_CONFIG_DISPLAY_NAMES: dict[tuple[str, str], str] = {
    ("graphsage_u", "topology"):   "GraphSAGE (topology)",
    ("graphsage_u", "attributes"): "GraphSAGE (+ attributes)",
}

# Canonical column / legend / x-axis order, matching Tables 10-11: baselines
# first, then GARG-AML base scores, then GARG-AML + tree, then + boost.
MODEL_ORDER: list[str] = list(MODEL_DISPLAY_NAMES.keys())


def pretty(key: str) -> str:
    """Return the canonical display name for ``key``.

    Unknown keys are returned unchanged so callers can pass through
    arbitrary labels (e.g. dataset names) safely.
    """
    return MODEL_DISPLAY_NAMES.get(key, key)


def pretty_config(key: str, config: str = "full") -> str:
    """Display name for model ``key`` trained on feature ``config``.

    ``config="full"`` reproduces :func:`pretty` exactly. A (key, config) pair
    in :data:`MODEL_CONFIG_DISPLAY_NAMES` wins over every rule below it, for
    models whose configs are their own thing rather than ablations. The
    ablations get a suffixed label, except ``topology``, which gets a name of
    its own for the reason documented on :data:`TOPOLOGY_DISPLAY_NAMES`.

    Unknown configs fall back to a parenthesised config name rather than
    raising: this is a labelling helper, and a missing table label should
    not take down a results run. Unknown *keys* pass through as
    :func:`pretty` already does.
    """
    if (key, config) in MODEL_CONFIG_DISPLAY_NAMES:
        return MODEL_CONFIG_DISPLAY_NAMES[(key, config)]
    if config == "topology":
        return TOPOLOGY_DISPLAY_NAMES.get(key, pretty(key) + " (topology only)")
    if config in FEATURE_CONFIG_LABELS:
        return pretty(key) + FEATURE_CONFIG_LABELS[config]
    return f"{pretty(key)} ({config})"


def pretty_many(keys: Iterable[str]) -> list[str]:
    """Vectorised :func:`pretty` for lists of keys."""
    return [pretty(k) for k in keys]


def gargaml_key(variant: str, directed: bool) -> str:
    """Build a canonical key for a GARG-AML variant.

    Parameters
    ----------
    variant : {"base", "tree", "boost"}
        Which GARG-AML model.
    directed : bool
        True for the directed score, False for the undirected score.

    Returns
    -------
    str
        Internal key, e.g. ``"gargaml_tree_u"``.
    """
    suffix = "d" if directed else "u"
    if variant == "base":
        return f"gargaml_{suffix}"
    if variant in {"tree", "boost"}:
        return f"gargaml_{variant}_{suffix}"
    raise ValueError(
        f"Unknown GARG-AML variant {variant!r}; expected 'base', 'tree' or 'boost'."
    )
