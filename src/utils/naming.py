"""
Canonical model naming for GARG-AML results.

Single source of truth for the display names used in tables, figures and
legends across the paper. The reviewer flagged inconsistencies between
Table 9 (lowercase, e.g. ``gargaml tree undirected``), Tables 10-11
(``GARG-AML Undir. Tree``) and Figures 8-9 (``GARG-AML Tree Undir.``);
this module fixes them by routing every label through one mapping.

Two-tier rule:

* **Internal keys** (CSV columns, filename suffixes, dict keys, the raw
  ``method`` column in dataframes) stay as the lowercase snake-case
  identifiers in ``MODEL_DISPLAY_NAMES`` so existing result files keep
  working.
* **Display names** (legends, axis labels, table headers, CD-diagram
  labels, written text) come from :func:`pretty`.

To add a new model (e.g. the topology-only baseline requested by the
reviewer), add one entry to ``MODEL_DISPLAY_NAMES`` and append the key
to ``MODEL_ORDER``.
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
    # Isolation-forest baseline (scripts/gargaml_IF.py): directed only -- it reads the
    # 9 directed block-density measures unconditionally, so there is no _u counterpart
    # to add until that script itself supports the undirected measure columns.
    "gargaml_if_d":    "GARG-AML Isolation Forest",
}

# Canonical column / legend / x-axis order. Keep this list aligned with
# the order used in Tables 10-11: baselines first, then GARG-AML base
# scores, then GARG-AML + tree, then GARG-AML + boost.
MODEL_ORDER: list[str] = list(MODEL_DISPLAY_NAMES.keys())


def pretty(key: str) -> str:
    """Return the canonical display name for ``key``.

    Unknown keys are returned unchanged so callers can pass through
    arbitrary labels (e.g. dataset names) safely.
    """
    return MODEL_DISPLAY_NAMES.get(key, key)


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
