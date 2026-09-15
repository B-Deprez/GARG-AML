"""
GraphSAGE baseline (task 1) -- minimal, tutorial-style first pass.

R2's biggest ask: the paper argues against "black-box GNNs" with none in the
comparison. A 2-layer GraphSAGE has the same receptive field as GARG-AML's
second-order neighbourhood, so it is the natural baseline. This first pass is
deliberately close to a plain PyTorch Geometric node-classification tutorial
rather than the full spec in GARG-AML_code_changes.md task 1: one feature
config (topology-only), NeighborLoader mini-batch training with no early
stopping, no timing instrumentation, HI-Small only. Those are fast-follows,
not silently dropped -- see the module docstring in scripts/graphsage_baseline.py.

The model must **never** see GARG-AML scores, block densities or block sizes --
that would destroy the point of comparing an interpretable hand-built score
against a GNN with the same receptive field.
"""

import os

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

from src.data.graph_construction import construct_IBM_graph

# GARG-AML's config-A "strict parity" features (in-degree, out-degree, log-degree,
# distinct counterparties) collapse to two numbers on this graph: parallel
# transactions between a pair are already collapsed into one edge by
# construct_IBM_graph, and the primary run is undirected, so in-degree ==
# out-degree == distinct-counterparties == plain degree.
NUM_FEATURES = 2


def build_graph_data(dataset, results_dir="results", cache=True):
    """Build (or load the cached) undirected PyG graph for ``dataset``.

    Returns ``(data, node_order)`` -- ``node_order[i]`` is the account for row
    ``i`` of every tensor on ``data``. Caching matters here: this graph is
    identical across every cutoff/target/fold this script trains on, and
    construct_IBM_graph + from_networkx over HI-Small's ~5M edges is not free.
    """
    cache_path = f"{results_dir}/{dataset}_graphsage_data.pt"
    if cache and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        return cached["data"], cached["node_order"]

    G = construct_IBM_graph(path=f"data/{dataset}_Trans.csv", directed=False)

    # Captured before from_networkx: it maps nodes to indices in this same
    # iteration order internally, so this is what keeps feature/label/fold
    # alignment correct without needing from_networkx to hand the mapping back.
    node_order = list(G.nodes())

    data = from_networkx(G)

    degree = np.array([d for _, d in G.degree()], dtype=np.float32)
    log_degree = np.log1p(degree)
    x = np.stack([degree, log_degree], axis=1)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
    data.x = torch.tensor(x, dtype=torch.float)

    if cache:
        torch.save({"data": data, "node_order": node_order}, cache_path)

    return data, node_order


class GraphSAGEModel(torch.nn.Module):
    """Plain 2-layer GraphSAGE, matching PyG's standard node-classification
    tutorial shape. Outputs one raw logit per node (binary classification via
    BCEWithLogitsLoss, not a per-class softmax)."""

    def __init__(self, in_channels, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_fold(data, train_mask, y, device, epochs=10, hidden_channels=64,
               dropout=0.2, lr=1e-3, batch_size=1024, num_neighbors=(25, 10),
               seed=1997):
    """Train one fresh GraphSAGEModel on ``train_mask``, transductively.

    The loader samples neighbourhoods from the *whole* graph (message passing
    sees every node, train and test alike -- this pipeline is transductive
    throughout, see CLAUDE.md); only the seed nodes drawn by
    ``input_nodes=train_mask`` -- and therefore the loss -- are restricted to
    the training fold. ``pos_weight`` handles the extreme class imbalance
    (BCEWithLogitsLoss, not resampling, per the task-1 spec) -- without it the
    model can trivially predict all-negative and the baseline would be
    meaningless.
    """
    torch.manual_seed(seed)

    # Reassigned in place each call (same convention as laundering_combined's
    # per-cutoff/target column reuse in gargaml_tree.py) -- cheaper than
    # cloning the whole graph (edge_index alone is ~2x HI-Small's ~5M edges)
    # every fold, and safe because training is sequential, not concurrent.
    data.y = torch.tensor(y, dtype=torch.float)

    loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=train_mask, shuffle=True,
    )

    model = GraphSAGEModel(data.num_node_features, hidden_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = int(y[train_mask].sum())
    n_neg = int(train_mask.sum()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = criterion(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * batch.batch_size
        print(f"    epoch {epoch}: loss {total_loss / max(int(train_mask.sum()), 1):.4f}")

    return model


def score_all(model, data, device):
    """Sigmoid score for every node, one full-graph forward pass.

    Training is mini-batched (train_fold) for scalability; inference is a
    single cheap forward pass under no_grad, the standard PyG shortcut that
    avoids a second loader.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
    return torch.sigmoid(logits).cpu().numpy()
