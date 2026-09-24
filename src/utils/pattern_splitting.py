"""
What the Louvain pre-processing destroys.

The severance figures say the pre-processing discards most of the graph's
edges. On its own that number settles nothing: it is either catastrophic or
irrelevant depending entirely on *which* edges go. This module measures, per
known laundering attempt, whether the structure GARG-AML looks for survives
the reduction.

Edge survival is not the number that matters
--------------------------------------------
GARG-AML does not key on how many edges a pattern keeps. It keys on the
**block contrast of a second-order neighbourhood**, which needs the two-hop
path -- source to mule to target -- to still be there. A pattern can keep
most of its edges and still lose every 2-path, and it is then invisible to
the score while looking barely touched by an edge count. So both are
reported, and ``path_survival`` is the one to read:

``edge_survival``
    fraction of the attempt's own edges whose endpoints stayed in one
    community.
``path_survival``
    fraction of the attempt's own directed 2-paths (``a -> b -> c``, all
    three inside the attempt) whose **both** legs survived.
``detectable_after``
    whether *any* 2-path survived. Once it is false, no resolution of the
    score can find that pattern, because the structure it looks for is gone.

Patterns with no 2-path to begin with
-------------------------------------
FAN-OUT and FAN-IN are one hop deep by construction, so ``paths_total`` is 0
and the path columns are NaN rather than 0 -- there is nothing to destroy,
and scoring them as "100 % destroyed" or "100 % survived" would both be
wrong. GARG-AML targets GATHER-SCATTER and SCATTER-GATHER, which are exactly
the two-hop shapes, so :func:`summarise` keeps the breakdown by type rather
than pooling.

The partition is the pipeline's own
-----------------------------------
:func:`analyse` takes a ``communities`` mapping from
:func:`src.utils.graph_processing.community_map`, which is the same Louvain
call, undirected view and seed that ``graph_community`` uses to build the
reduced graph. Re-deriving a partition here would answer a question about a
graph nobody scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Instance-level columns, in the order the output CSV carries them.
INSTANCE_COLUMNS = [
    "dataset", "resolution", "instance", "pattern_type",
    "n_accounts", "n_in_graph", "n_communities", "split",
    "edges_total", "edges_kept", "edge_survival",
    "paths_total", "paths_kept", "path_survival",
    "detectable_before", "detectable_after",
]


def instance_edges(instances):
    """``{instance: (pattern_type, [(source, target), ...])}``, deduplicated.

    An attempt is a set of *transactions*, and several of them can run
    between the same pair of accounts. The graph collapses parallel
    transactions into one edge, so the diagnostic has to as well -- counting
    transactions would weight an attempt by how often its accounts
    transacted rather than by its structure.
    """
    grouped = {}
    for instance, rows in instances.groupby("instance"):
        pattern_type = rows["pattern_type"].iloc[0]
        edges = sorted({(s, t) for s, t in zip(rows["source"], rows["target"])
                        if s != t})  # self-loops are dropped from the graph too
        grouped[int(instance)] = (pattern_type, edges)
    return grouped


def two_paths(edges):
    """Directed 2-paths ``a -> b -> c`` within ``edges``, with ``a != c``.

    ``a == c`` is excluded because a reciprocal pair is not a money-flow
    path through a mule: it is the same two accounts paying each other back,
    which is the opposite of the structure being looked for.
    """
    out = {}
    for a, b in edges:
        out.setdefault(a, []).append(b)

    paths = []
    for a, b in edges:
        for c in out.get(b, ()):
            if c != a:
                paths.append((a, b, c))
    return paths


def analyse_instance(pattern_type, edges, nodes, communities):
    """One attempt's survival under a given partition.

    ``communities`` maps node to community index; a node absent from it is
    absent from the graph and its edges cannot survive. ``nodes`` is the
    graph's node set, used to separate "this account is not in the graph"
    from "this account is in a different community".
    """
    accounts = sorted({n for edge in edges for n in edge})
    in_graph = [a for a in accounts if a in nodes]

    def kept(u, v):
        return (u in communities and v in communities
                and communities[u] == communities[v])

    edges_kept = [e for e in edges if kept(*e)]

    paths = two_paths(edges)
    paths_kept = [p for p in paths if kept(p[0], p[1]) and kept(p[1], p[2])]

    n_communities = len({communities[a] for a in in_graph if a in communities})

    return {
        "pattern_type": pattern_type,
        "n_accounts": len(accounts),
        "n_in_graph": len(in_graph),
        "n_communities": n_communities,
        # "Split" means the attempt's accounts did not all land together.
        # Exactly one community is intact; zero means none of its accounts
        # reached the graph at all, which is not a split and is flagged by
        # n_in_graph rather than hidden in this boolean.
        "split": bool(n_communities > 1),
        "edges_total": len(edges),
        "edges_kept": len(edges_kept),
        "edge_survival": len(edges_kept) / len(edges) if edges else np.nan,
        "paths_total": len(paths),
        "paths_kept": len(paths_kept),
        # NaN, not 0: a one-hop pattern (FAN-OUT, FAN-IN) has no 2-path to
        # lose, and reporting it as fully destroyed would be a fabrication.
        "path_survival": len(paths_kept) / len(paths) if paths else np.nan,
        "detectable_before": bool(paths),
        "detectable_after": bool(paths_kept),
    }


def analyse(instances, nodes, communities, dataset, resolution):
    """Every attempt in ``instances``, as one row each."""
    rows = []
    for instance, (pattern_type, edges) in instance_edges(instances).items():
        row = analyse_instance(pattern_type, edges, nodes, communities)
        row.update(dataset=dataset, instance=instance,
                   resolution="off" if resolution is None else resolution)
        rows.append(row)

    frame = pd.DataFrame(rows)
    return frame[[c for c in INSTANCE_COLUMNS if c in frame.columns]]


def resolution_sort_key(value):
    """Sort key putting the no-Louvain control first, then ascending resolution.

    ``resolution`` holds ``"off"`` beside numbers, so the column is object
    dtype and both ``groupby`` and a plain sort order it as text, which puts
    10 and 20 before 5. The axis a reader follows is how much the setting
    reduces the graph.
    """
    return (0, 0.0) if value == "off" else (1, float(value))


def summarise(frame):
    """Per (dataset, resolution, pattern type): how much survived.

    ``pct_destroyed`` is the headline: the share of attempts that had a
    two-hop structure and no longer do. It is computed over the attempts
    that *had* one, so a one-hop pattern type contributes no denominator
    rather than a misleading zero.
    """
    keys = ["dataset", "resolution", "pattern_type"]

    def _agg(group):
        had = group[group["detectable_before"]]
        return pd.Series({
            "attempts": len(group),
            "pct_split": 100.0 * group["split"].mean(),
            "mean_communities": group["n_communities"].mean(),
            "mean_edge_survival": 100.0 * group["edge_survival"].mean(),
            "attempts_with_2path": len(had),
            "mean_path_survival": (100.0 * had["path_survival"].mean()
                                   if len(had) else np.nan),
            "pct_destroyed": (100.0 * (~had["detectable_after"]).mean()
                              if len(had) else np.nan),
        })

    summary = frame.groupby(keys, dropna=False).apply(_agg).reset_index()

    totals = frame.groupby(keys[:2], dropna=False).apply(_agg).reset_index()
    totals["pattern_type"] = "ALL"

    out = pd.concat([summary, totals], ignore_index=True)[
        keys + ["attempts", "pct_split", "mean_communities",
                "mean_edge_survival", "attempts_with_2path",
                "mean_path_survival", "pct_destroyed"]]

    # "ALL" last within each resolution; resolutions in sweep order.
    out["_res"] = out["resolution"].map(resolution_sort_key)
    out["_type"] = out["pattern_type"] == "ALL"
    return (out.sort_values(["dataset", "_res", "_type", "pattern_type"])
               .drop(columns=["_res", "_type"]).reset_index(drop=True))
