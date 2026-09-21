import os
import re

import pandas as pd
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Louvain sensitivity (task 4)
# ---------------------------------------------------------------------------
# R2-M3: resolution = 10 is unjustified, and dropping every inter-community
# edge alters the neighbourhoods being scored -- a pattern straddling two
# communities is destroyed before the score sees it.
#
# The sweep reuses task 5's trick rather than adding a parameter to every
# script: **the setting is part of the dataset name**. "HI-Small_res20" is
# HI-Small pre-processed at resolution 20, "HI-Small_nolouvain" is HI-Small
# with the reduction switched off, and a bare "HI-Small" keeps the published
# default. Since every output path in this repository is built from the
# dataset string, each setting writes its own measures, metrics and tables
# with no further plumbing, and the published filenames stay untouched.
#
# The token may sit anywhere in the name, so it composes with a bank view in
# either order ("HI-Small_res20_bank012" and "HI-Small_bank012_res20" both
# work) -- parse_view partitions on the first "_bank", so a suffix-only rule
# would have mis-parsed one of the two.
RESOLUTION_SEPARATOR = "res"
NO_LOUVAIN = "nolouvain"
DEFAULT_RESOLUTION = 10  # the published value; see graph_community below

_RESOLUTION_TOKEN = re.compile(r"^" + RESOLUTION_SEPARATOR + r"(\d+(?:\.\d+)?)$")


def parse_resolution(name, default=DEFAULT_RESOLUTION):
    """``name`` -> ``(name without the token, resolution)``.

    ``resolution`` is ``None`` when the name asks for no Louvain at all.
    A name carrying no token is returned unchanged with ``default``, which
    is what keeps every existing dataset on the published setting.

    >>> parse_resolution("HI-Small")
    ('HI-Small', 10)
    >>> parse_resolution("HI-Small_res20")
    ('HI-Small', 20.0)
    >>> parse_resolution("HI-Small_nolouvain")
    ('HI-Small', None)
    """
    parts = name.split("_")
    kept, resolution = [], default
    for part in parts:
        if part == NO_LOUVAIN:
            resolution = None
            continue
        match = _RESOLUTION_TOKEN.match(part)
        if match:
            resolution = float(match.group(1))
            continue
        kept.append(part)
    return "_".join(kept), resolution


def strip_resolution(name):
    """``name`` with any Louvain token removed; see :func:`parse_resolution`."""
    return parse_resolution(name)[0]


def reduce_graph(G, resolution=DEFAULT_RESOLUTION, dataset=None,
                 results_dir="results"):
    """Apply the Louvain pre-processing at ``resolution``, or not at all.

    ``resolution=None`` returns ``G`` unchanged. That is the no-Louvain arm
    of task 4 -- the upper bound on what the pre-processing costs in
    detection performance -- and it is a genuine identity, not a resolution
    so low that everything lands in one community: Louvain is never run, so
    no edge is dropped and no seed matters.

    **The no-Louvain arm is expensive, and not linearly so.** The reduction
    is what keeps a second-order ego graph small; without it HI-Small's
    reach ~14,900 accounts, and `GARG_AML_node_*_measures` densifies each
    one with `nx.adjacency_matrix(...).toarray()` -- about 1.8 GB for a
    single node (measured while building task 5's appendix, which is why
    that experiment scores only a bank's own clients). Budget for it, and
    expect LI-Large's no-Louvain arm to be infeasible rather than merely
    slow.

    When ``dataset`` is given, one row of severance statistics is appended
    to ``results/louvain_severance.csv``: the % of edges dropped is exactly
    what R2-M3 and the carried-over JMLC point ask to see per setting, and
    logging it here means every run records it without the measure scripts
    having to.
    """
    H = G if resolution is None else graph_community(G, resolution=resolution)
    _log_severance(dataset, resolution, G, H, results_dir)
    return H


def _log_severance(dataset, resolution, G, H, results_dir="results"):
    """Print the edge-severance headline; append a row when ``dataset`` is given.

    Printing is unconditional because it costs nothing and the number is the
    point of the sweep. The CSV append is not: stage 1 (the measure scripts)
    and stage 2 (gargaml_tree.py) both build this graph, so logging from
    both would double every row. Stage 1 passes ``dataset`` and owns the
    record; stage 2 leaves it ``None`` and only prints.
    """
    before, after = G.number_of_edges(), H.number_of_edges()
    severed = before - after
    pct = 100.0 * severed / before if before else float("nan")

    print(f"Louvain resolution={resolution if resolution is not None else 'off'}: "
          f"{severed:,} of {before:,} edges severed ({pct:.2f}%)")

    if dataset is None:
        return

    row = pd.DataFrame([{
        "dataset": dataset,
        "resolution": "off" if resolution is None else resolution,
        "nodes": G.number_of_nodes(),
        "edges_before": before,
        "edges_after": after,
        "edges_severed": severed,
        "pct_severed": pct,
    }])

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "louvain_severance.csv")
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def graph_degree(G, degree_cutoff=0.01):
    # Delete the hubs
    # The cut-off is defined as a relative number
    G_copy = G.copy()
    
    degree_df = pd.DataFrame(
        dict(
            G_copy.degree()
        ), 
        index = ["Degree"]
    ).transpose()

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

    Extracted from :func:`graph_community` so the task-4 pattern-splitting
    diagnostic can ask which community an account landed in **without
    reimplementing the partition**. That matters more than it looks: the
    diagnostic's whole claim is about the edges the pipeline drops, so a
    second Louvain call with a different undirected view or a different
    seed would answer a question about a partition nobody scored.
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

    # Create a new graph with only intra-community edges
    if directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
        
    H.add_nodes_from(G.nodes(data=True))  # Add all nodes with their attributes

    # Add only edges that connect nodes within the same community
    for u, v in G.edges():
        if node_community[u] == node_community[v]:
            H.add_edge(u, v, **G[u][v])
        
    return(H)