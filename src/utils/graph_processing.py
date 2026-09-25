import os
import re

import pandas as pd
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Pre-processing sensitivity
# ---------------------------------------------------------------------------
# The resolution sweep encodes **the setting in the dataset name** rather than
# adding a parameter to every script: "HI-Small_res20" is resolution 20,
# "HI-Small_nolouvain" has the reduction switched off, a bare "HI-Small" takes
# the default. Every output path is built from the dataset string, so each
# setting writes its own measures/metrics/tables with no further plumbing.
# The token may sit anywhere in the name, so it composes with a bank view in
# either order ("HI-Small_res20_bank012" or "HI-Small_bank012_res20"), since
# parse_view partitions on the first "_bank".
#
# "HI-Small_hubs100" is the other pre-processing arm: no Louvain, the 100
# highest-degree accounts removed instead. It carries no resolution of its
# own, so the token implies Louvain off unless a "_res<r>" is also given.
RESOLUTION_SEPARATOR = "res"
NO_LOUVAIN = "nolouvain"
HUBS_SEPARATOR = "hubs"
DEFAULT_RESOLUTION = 10  # the value the paper reports; see graph_community

_RESOLUTION_TOKEN = re.compile(r"^" + RESOLUTION_SEPARATOR + r"(\d+(?:\.\d+)?)$")
_HUBS_TOKEN = re.compile(r"^" + HUBS_SEPARATOR + r"(\d+)$")


def _parse_setting(name, default=DEFAULT_RESOLUTION):
    """``name`` -> ``(name without the tokens, resolution, hubs)``."""
    kept, resolution, hubs, explicit = [], default, None, False
    for part in name.split("_"):
        if part == NO_LOUVAIN:
            resolution, explicit = None, True
            continue
        match = _RESOLUTION_TOKEN.match(part)
        if match:
            resolution, explicit = float(match.group(1)), True
            continue
        match = _HUBS_TOKEN.match(part)
        if match:
            hubs = int(match.group(1))
            continue
        kept.append(part)
    if hubs is not None and not explicit:
        resolution = None  # hub removal replaces the Louvain step
    return "_".join(kept), resolution, hubs


def parse_resolution(name, default=DEFAULT_RESOLUTION):
    """``name`` -> ``(name without the token, resolution)``.

    ``resolution`` is ``None`` when the name asks for no Louvain at all.
    A name carrying no token is returned unchanged with ``default``.

    >>> parse_resolution("HI-Small")
    ('HI-Small', 10)
    >>> parse_resolution("HI-Small_res20")
    ('HI-Small', 20.0)
    >>> parse_resolution("HI-Small_nolouvain")
    ('HI-Small', None)
    >>> parse_resolution("HI-Small_hubs100")
    ('HI-Small', None)
    """
    base, resolution, _ = _parse_setting(name, default)
    return base, resolution


def parse_hubs(name):
    """``name`` -> the number of hubs to remove, or ``None``.

    >>> parse_hubs("HI-Small_hubs100")
    100
    >>> parse_hubs("HI-Small_res20") is None
    True
    """
    return _parse_setting(name)[2]


def setting_label(resolution, hubs=None):
    """The pre-processing arm as one value: ``"off"``, ``"hubs5"`` or ``10.0``.

    Shared by the severance log and the reporting tables so an arm is named
    the same everywhere.
    """
    if hubs is not None:
        return HUBS_SEPARATOR + str(hubs)
    return "off" if resolution is None else resolution


def strip_resolution(name):
    """``name`` with any pre-processing token removed; see :func:`parse_resolution`."""
    return parse_resolution(name)[0]


def reduce_graph(G, resolution=DEFAULT_RESOLUTION, dataset=None,
                 results_dir="results", hubs=None):
    """Apply the Louvain pre-processing at ``resolution``, or not at all.

    ``hubs=k`` is the other arm of the sensitivity analysis: Louvain is
    skipped and the ``k`` highest-degree accounts are removed instead (see
    :func:`graph_degree`). It is not combined with a resolution.

    ``resolution=None`` returns ``G`` unchanged: the no-Louvain arm of the
    sweep, which bounds what the pre-processing costs in detection
    performance. A genuine identity, not a resolution so low that everything
    lands in one community -- Louvain is never run, so no edge is dropped.

    **The no-Louvain arm is expensive, and not linearly so.** The reduction
    is what keeps a second-order ego graph small; without it a HI-Small ego
    graph reaches tens of thousands of accounts, and
    ``GARG_AML_node_*_measures`` densifies each one with
    ``nx.adjacency_matrix(...).toarray()``. Expect LI-Large's no-Louvain arm
    to be infeasible rather than merely slow.

    When ``dataset`` is given, one row of severance statistics is appended to
    ``results/louvain_severance.csv``, so every run records the share of
    edges dropped without the measure scripts having to.
    """
    if hubs is not None:
        H = graph_degree(G, n_hubs=hubs)
    else:
        H = G if resolution is None else graph_community(G, resolution=resolution)
    _log_severance(dataset, setting_label(resolution, hubs), G, H, results_dir)
    return H


def _log_severance(dataset, setting, G, H, results_dir="results"):
    """Print the edge-severance headline; append a row when ``dataset`` is given.

    The CSV append is conditional because stage 1 (the measure scripts) and
    stage 2 (gargaml_tree.py) both build this graph, so logging from both
    would double every row. Stage 1 passes ``dataset`` and owns the record;
    stage 2 leaves it ``None`` and only prints.
    """
    before, after = G.number_of_edges(), H.number_of_edges()
    severed = before - after
    pct = 100.0 * severed / before if before else float("nan")

    print(f"Pre-processing {setting}: "
          f"{severed:,} of {before:,} edges severed ({pct:.2f}%)")

    if dataset is None:
        return

    row = pd.DataFrame([{
        "dataset": dataset,
        "resolution": setting,
        "nodes": G.number_of_nodes(),
        "edges_before": before,
        "edges_after": after,
        "edges_severed": severed,
        "pct_severed": pct,
    }])

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "louvain_severance.csv")
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def graph_degree(G, degree_cutoff=0.01, n_hubs=None):
    # Hub removal: degree_cutoff is a relative top fraction; n_hubs, when
    # given, removes exactly that many highest-degree nodes instead.
    G_copy = G.copy()

    # Ranked on the undirected view, as community_map's partition is, so the
    # directed and undirected runs of an arm remove the same accounts.
    view = G.to_undirected(as_view=True) if nx.is_directed(G) else G
    degree_df = pd.DataFrame(
        dict(
            view.degree()
        ), 
        index = ["Degree"]
    ).transpose()

    if n_hubs is not None:
        # A boundary tie is broken by node order (the CSV's), so reproducible.
        hubs_deleted = list(degree_df["Degree"].nlargest(n_hubs).index)
    else:
        degree_threshold = degree_df["Degree"].quantile(1 - degree_cutoff)
        hub_criteria = degree_df["Degree"] >= degree_threshold

        hubs_deleted = list(
                    degree_df[hub_criteria].reset_index()["index"]
                )

    G_copy.remove_nodes_from(
        hubs_deleted
    )        
    
    return(G_copy)

def community_map(G, resolution=DEFAULT_RESOLUTION):
    """``node -> community index``, from the Louvain call the pipeline uses.

    Shared with :func:`graph_community` so the pattern-splitting diagnostic
    can ask which community an account landed in **without reimplementing
    the partition**: its claim is about the edges the pipeline drops, and a
    second Louvain call with a different undirected view or seed would
    answer a question about a partition nobody scored.
    """
    G_undirected = G.copy().to_undirected() if nx.is_directed(G) else G.copy()

    community_list = nx.community.louvain_communities(
        G_undirected, resolution=resolution, seed=1997)

    node_community = {}
    for idx, community in enumerate(community_list):
        for node in community:
            node_community[node] = idx
    return node_community


def graph_community(G, resolution=10): # large resolution to have smaller communities
    directed = nx.is_directed(G)

    node_community = community_map(G, resolution)

    # A new graph keeping every node, but only intra-community edges.
    if directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()

    H.add_nodes_from(G.nodes(data=True))

    for u, v in G.edges():
        if node_community[u] == node_community[v]:
            H.add_edge(u, v, **G[u][v])
        
    return(H)