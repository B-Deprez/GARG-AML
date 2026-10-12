import pandas as pd
import networkx as nx

from src.data.bank_views import BANK_COLUMNS, filter_transactions, resolve_banks

def construct_IBM_graph(path="data/HI-Small_Trans.csv", directed=False, banks=None):
    """
    Construct a graph from the IBM data.

    ``banks`` restricts the graph to a single institution's view: only
    transactions booked at one of those banks are kept, i.e. those with at
    least one endpoint among its clients. ``None`` is the full graph.
    """
    banks = resolve_banks(banks, path)  # expands a group spec such as "top50"

    # Bank identifiers are zero-padded and infer as int64 ("010" -> 10), so
    # they're forced to str only when a view is being built. Account columns
    # keep their inferred dtype -- they're the node keys, and reading them
    # differently would change node identity.
    dtype = {c: str for c in BANK_COLUMNS} if banks is not None else None

    data = pd.read_csv(path, dtype=dtype)
    data = filter_transactions(data, banks)

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    edges = zip(data["Account"], data["Account.1"])

    G.add_edges_from(edges)
    G.remove_edges_from([(n, n) for n in G.nodes() if G.has_edge(n, n)])

    return G

def construct_synthetic_graph(path="data/edge_data_synthetic.csv", directed=False):
    """
    Construct a graph from the synthetic data.
    """
    data = pd.read_csv(path)

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    edges = zip(data["source"], data["target"])

    G.add_edges_from(edges)
    G.remove_edges_from([(n, n) for n in G.nodes() if G.has_edge(n, n)])

    return G