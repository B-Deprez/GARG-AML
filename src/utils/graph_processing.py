import os
import re

import pandas as pd
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Louvain sensitivity
# ---------------------------------------------------------------------------
# The resolution sweep encodes **the setting in the dataset name** rather than
# adding a parameter to every script. "HI-Small_res20" is HI-Small
# pre-processed at resolution 20, "HI-Small_nolouvain" is HI-Small with the
# reduction switched off, and a bare "HI-Small" takes the default. Since every
# output path in this repository is built from the dataset string, each
# setting writes its own measures, metrics and tables with no further
# plumbing.
#
# The token may sit anywhere in the name, so it composes with a bank view in
# either order: "HI-Small_res20_bank012" and "HI-Small_bank012_res20" both
# parse, because parse_view partitions on the first "_bank".
RESOLUTION_SEPARATOR = "res"
NO_LOUVAIN = "nolouvain"
DEFAULT_RESOLUTION = 10  # the value the paper reports; see graph_community

_RESOLUTION_TOKEN = re.compile(r"^" + RESOLUTION_SEPARATOR + r"(\d+(?:\.\d+)?)$")


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

    ``resolution=None`` returns ``G`` unchanged: the no-Louvain arm of the
    sweep, which bounds what the pre-processing costs in detection
    performance. It is a genuine identity rather than a resolution so low
    that everything lands in one community -- Louvain is never run, so no
    edge is dropped and no seed matters.

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
    H = G if resolution is None else graph_community(G, resolution=resolution)
    _log_severance(dataset, resolution, G, H, results_dir)
    return H


def _log_severance(dataset, resolution, G, H, results_dir="results"):
    """Print the edge-severance headline; append a row when ``dataset`` is given.

    The CSV append is conditional because stage 1 (the measure scripts) and
    stage 2 (gargaml_tree.py) both build this graph, so logging from both
    would double every row. Stage 1 passes ``dataset`` and owns the record;
    stage 2 leaves it ``None`` and only prints.
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
    # Hub removal; degree_cutoff is relative, the top fraction by degree.
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