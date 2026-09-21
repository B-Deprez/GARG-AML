"""
GraphSAGE baseline (task 1).

R2's largest ask: the paper argues against "black-box GNNs" while comparing
against none. A **2-layer** GraphSAGE has the same receptive field as
GARG-AML's second-order neighbourhood, so it is the principled baseline --
same information, one model interpretable and O(|V|), the other needing a GPU
and neighbour sampling. That framing is the reason for every choice below.

The model must **never** see GARG-AML scores, block densities or block sizes.
Feeding them in would collapse the comparison into "GARG-AML plus a GNN",
which answers no question anyone asked. Both feature configurations here are
built from the raw transaction file alone.

Feature configurations
----------------------
``topology`` (config A, strict parity)
    Degree and log-degree. The spec lists four items -- in-degree,
    out-degree, log-degree, distinct counterparties -- but on this graph they
    collapse to two: parallel transactions between a pair are merged into one
    edge and the primary run is undirected, so in-degree == out-degree ==
    distinct counterparties == degree. Emitting four columns would be three
    copies of one number wearing different labels.

``attributes`` (config B, deliberately generous)
    Config A plus per-account amount, count, currency, bank and timing
    aggregates. Deliberately generous: if GARG-AML holds up against a GNN
    that also sees transaction amounts, that is the strongest sentence
    available in the paper.

Caching
-------
Three separate caches, because they have different lifetimes: the graph
structure (identical for every run), the feature matrix (per config), and
training checkpoints (per fold). The structure and features are what make a
sweep affordable -- they are built once and reused across every cut-off,
target and fold.

Scale
-----
The ``pandas`` structure backend builds ``edge_index`` straight from the CSV
columns without materialising a NetworkX graph, which is what makes LI-Large
(176M edges) plausible; the ``networkx`` backend goes through
``construct_IBM_graph`` and is kept because it shares its node universe with
the rest of the pipeline by construction. Both are verified to produce the
same node set and edge count on HI-Small.
"""

import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

from src.data.graph_construction import construct_IBM_graph
from src.data.bank_views import (BANK_COLUMNS, filter_transactions,
                                parse_view, resolve_banks, trans_path)

SEED = 1997

# Account and bank identifiers are read as strings, matching
# define_ML_labels: "00123" and 123 are different accounts to one and the
# same account to the other, and the label table is joined on these values.
ID_DTYPES = {"From Bank": str, "To Bank": str, "Account": str, "Account.1": str}

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

# Column groups per feature config. The names are the schema written for the
# appendix (task 10), so they are part of the published output.
TOPOLOGY_FEATURES = ["degree", "log_degree"]

ATTRIBUTE_FEATURES = [
    "n_sent", "n_received",
    "amount_out_sum", "amount_out_mean", "amount_out_std", "amount_out_max",
    "amount_in_sum", "amount_in_mean", "amount_in_std", "amount_in_max",
    "n_currencies", "n_counterparty_banks",
    "active_span_days", "mean_gap_hours",
]

FEATURE_CONFIGS = {
    "topology": TOPOLOGY_FEATURES,
    "attributes": TOPOLOGY_FEATURES + ATTRIBUTE_FEATURES,
}

# Columns that are already counts or non-negative money amounts: log1p before
# standardising, or a handful of hub accounts dominate every feature and the
# z-scores of everyone else collapse into a spike at zero.
LOG_SCALED_FEATURES = {
    "degree", "n_sent", "n_received",
    "amount_out_sum", "amount_out_mean", "amount_out_std", "amount_out_max",
    "amount_in_sum", "amount_in_mean", "amount_in_std", "amount_in_max",
    "active_span_days", "mean_gap_hours",
}


def check_config(config):
    if config not in FEATURE_CONFIGS:
        raise KeyError("Unknown feature config "+repr(config)+"; expected one of "
                       +str(list(FEATURE_CONFIGS)))


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

def _structure_from_pandas(path, banks=None):
    """Undirected ``edge_index`` and node order, straight from the CSV.

    Never builds a Python graph object. On LI-Large that is the difference
    between an int64 tensor and hundreds of millions of dict entries, and it
    is why this is the default backend.

    ``banks`` restricts the structure to a single-bank view (task 5); the
    bank columns are only read when one is asked for.
    """
    usecols = ["Account", "Account.1"]
    if banks is not None:
        usecols = list(BANK_COLUMNS) + usecols
    frame = pd.read_csv(path, usecols=usecols, dtype=ID_DTYPES)
    frame = filter_transactions(frame, banks)

    # One shared categorical over both endpoint columns, so a node has the
    # same index whether it appears as sender or receiver.
    accounts = pd.Index(pd.unique(pd.concat([frame["Account"], frame["Account.1"]],
                                            ignore_index=True)))
    source = accounts.get_indexer(frame["Account"]).astype(np.int64)
    target = accounts.get_indexer(frame["Account.1"]).astype(np.int64)

    keep = source != target  # construct_IBM_graph drops self-loops; match it
    source, target = source[keep], target[keep]

    # Collapse parallel transactions, then symmetrise -- the same graph
    # nx.Graph gives, without the intermediate object.
    pairs = np.unique(np.stack([np.minimum(source, target),
                                np.maximum(source, target)], axis=1), axis=0)
    edge_index = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T

    return edge_index, list(accounts)


def _structure_from_networkx(path, banks=None):
    """Same structure via ``construct_IBM_graph``, for cross-checking."""
    G = construct_IBM_graph(path=path, directed=False, banks=banks)
    # Captured before from_networkx: it indexes nodes in this iteration
    # order internally, and that order is what aligns labels and folds.
    node_order = list(G.nodes())
    data = from_networkx(G)
    return data.edge_index.numpy(), node_order


def build_graph_structure(dataset, results_dir="results", backend="pandas", cache=True):
    """Build (or load) the undirected graph structure for ``dataset``.

    Returns ``(edge_index, node_order, degree)``. ``node_order[i]`` is the
    account at row ``i`` of every tensor built from this structure --
    features, labels and fold assignments are all aligned through it.

    ``dataset`` may be a task-5 view name ("HI-Small_bank012"), in which case
    the structure is built from that bank's visible transactions only and is
    cached under the view's own name. Note that ``node_order`` then holds
    every account in the view, including external counterparties: they carry
    messages through the graph but are not part of the evaluated population
    (see ``scripts/graphsage_baseline.py``).
    """
    cache_path = results_dir+"/"+dataset+"_graphsage_structure.pt"
    if cache and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        return cached["edge_index"], cached["node_order"], cached["degree"]

    path = trans_path(dataset)
    _, banks = parse_view(dataset)
    banks = resolve_banks(banks, path)
    if backend == "pandas":
        edge_index, node_order = _structure_from_pandas(path, banks=banks)
    elif backend == "networkx":
        edge_index, node_order = _structure_from_networkx(path, banks=banks)
    else:
        raise ValueError("Unknown backend "+repr(backend))

    # Each undirected edge appears twice in edge_index, so a row count over
    # the source column is exactly the undirected degree.
    degree = np.bincount(edge_index[0], minlength=len(node_order)).astype(np.float32)

    if cache:
        torch.save({"edge_index": edge_index, "node_order": node_order,
                    "degree": degree}, cache_path)

    return edge_index, node_order, degree


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class _AccountAggregator:
    """Streaming per-account aggregates over the transaction file.

    Chunked because LI-Large's CSV is ~16 GB. Everything here is a
    *combinable* statistic: counts, sums, sums of squares (which give std
    without a second pass), minima and maxima, plus deduplicated
    (account, currency) and (account, bank) pairs for the two nunique
    columns. Nothing requires holding the file in memory at once.
    """

    def __init__(self):
        self.sums = {}      # role -> DataFrame indexed by account
        self.pairs = {}     # name -> DataFrame of unique (account, value)
        self.times = {}     # role -> DataFrame of per-account min/max

    def _accumulate_sums(self, role, frame):
        grouped = frame.groupby("account").agg(
            n=("amount", "size"),
            total=("amount", "sum"),
            total_sq=("amount_sq", "sum"),
            maximum=("amount", "max"),
        )
        if role in self.sums:
            combined = pd.concat([self.sums[role], grouped])
            grouped = combined.groupby(level=0).agg(
                n=("n", "sum"), total=("total", "sum"),
                total_sq=("total_sq", "sum"), maximum=("maximum", "max"),
            )
        self.sums[role] = grouped

    def _accumulate_pairs(self, name, frame):
        unique = frame.drop_duplicates()
        if name in self.pairs:
            unique = pd.concat([self.pairs[name], unique]).drop_duplicates()
        self.pairs[name] = unique

    def _accumulate_times(self, frame):
        grouped = frame.groupby("account")["timestamp"].agg(["min", "max"])
        if "all" in self.times:
            combined = pd.concat([self.times["all"], grouped])
            grouped = combined.groupby(level=0).agg(min=("min", "min"), max=("max", "max"))
        self.times["all"] = grouped

    def add_chunk(self, chunk):
        chunk = chunk.copy()
        chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], format=TIMESTAMP_FORMAT,
                                            errors="coerce")

        for role, account_col, amount_col in [("out", "Account", "Amount Paid"),
                                              ("in", "Account.1", "Amount Received")]:
            side = pd.DataFrame({
                "account": chunk[account_col],
                "amount": chunk[amount_col].astype(np.float64),
            })
            side["amount_sq"] = side["amount"] ** 2
            self._accumulate_sums(role, side)

        # An account's own bank is near-constant; what varies -- and what a
        # model could plausibly use -- is the set of banks it deals *with*.
        self._accumulate_pairs("bank", pd.concat([
            pd.DataFrame({"account": chunk["Account"], "value": chunk["To Bank"]}),
            pd.DataFrame({"account": chunk["Account.1"], "value": chunk["From Bank"]}),
        ], ignore_index=True))

        self._accumulate_pairs("currency", pd.concat([
            pd.DataFrame({"account": chunk["Account"], "value": chunk["Payment Currency"]}),
            pd.DataFrame({"account": chunk["Account.1"], "value": chunk["Receiving Currency"]}),
        ], ignore_index=True))

        self._accumulate_times(pd.concat([
            pd.DataFrame({"account": chunk["Account"], "timestamp": chunk["Timestamp"]}),
            pd.DataFrame({"account": chunk["Account.1"], "timestamp": chunk["Timestamp"]}),
        ], ignore_index=True))

    def frame(self, node_order):
        """Assemble the aggregates into one row per account in ``node_order``."""
        index = pd.Index(node_order, name="account")
        out = pd.DataFrame(index=index)

        for role, prefix in [("out", "amount_out"), ("in", "amount_in")]:
            sums = self.sums[role].reindex(index).fillna(0.0)
            n = sums["n"]
            mean = np.where(n > 0, sums["total"] / n.replace(0, np.nan), 0.0)
            # Population variance from the running sums; clipped because
            # catastrophic cancellation can push it a hair below zero.
            variance = np.where(n > 0, sums["total_sq"] / n.replace(0, np.nan) - mean ** 2, 0.0)
            out["n_"+("sent" if role == "out" else "received")] = n.values
            out[prefix+"_sum"] = sums["total"].values
            out[prefix+"_mean"] = np.nan_to_num(mean)
            out[prefix+"_std"] = np.sqrt(np.clip(np.nan_to_num(variance), 0, None))
            out[prefix+"_max"] = sums["maximum"].values

        for name, column in [("currency", "n_currencies"), ("bank", "n_counterparty_banks")]:
            counts = self.pairs[name].groupby("account").size()
            out[column] = counts.reindex(index).fillna(0).values

        times = self.times["all"].reindex(index)
        span_days = (times["max"] - times["min"]).dt.total_seconds() / 86400.0
        out["active_span_days"] = np.nan_to_num(span_days.values)

        # Mean gap between an account's transactions. Zero for accounts with a
        # single transaction, which is the honest value for "no gap observed"
        # and keeps the column free of NaNs the model cannot consume.
        n_total = out["n_sent"].values + out["n_received"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(n_total > 1, out["active_span_days"].values * 24.0 / (n_total - 1), 0.0)
        out["mean_gap_hours"] = np.nan_to_num(gap)

        return out


def build_attribute_features(dataset, node_order, chunksize=2_000_000):
    """Per-account amount/count/currency/bank/timing aggregates (config B)."""
    path = trans_path(dataset)
    _, banks = parse_view(dataset)  # task 5: aggregate only what the bank sees
    banks = resolve_banks(banks, path)
    columns = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
               "Amount Received", "Receiving Currency", "Amount Paid",
               "Payment Currency"]

    aggregator = _AccountAggregator()
    for chunk in pd.read_csv(path, usecols=columns, dtype=ID_DTYPES, chunksize=chunksize):
        aggregator.add_chunk(filter_transactions(chunk, banks))

    return aggregator.frame(node_order)


def standardise(frame):
    """log1p the skewed columns, then z-score every column."""
    values = frame.copy()
    for column in values.columns:
        if column in LOG_SCALED_FEATURES:
            values[column] = np.log1p(np.clip(values[column].values, 0, None))
    x = values.values.astype(np.float32)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def build_features(dataset, node_order, degree, config, results_dir="results", cache=True):
    """Feature matrix for ``config``, aligned to ``node_order``.

    Cached per config: config B reads the full transaction file, which is the
    expensive part of a sweep and identical across every cut-off and fold.
    """
    check_config(config)
    cache_path = results_dir+"/"+dataset+"_graphsage_x_"+config+".pt"
    if cache and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    frame = pd.DataFrame({"degree": degree, "log_degree": np.log1p(degree)},
                         index=pd.Index(node_order, name="account"))

    if config == "attributes":
        frame = frame.join(build_attribute_features(dataset, node_order))

    frame = frame[FEATURE_CONFIGS[config]]
    x = standardise(frame)

    if cache:
        torch.save(x, cache_path)

    return x


def feature_schema(config):
    """The exact feature matrix of ``config`` as a documented table (task 10).

    Same shape as src/utils/features.py's schema, so the GNN baseline's
    appendix row is built from the same columns as the tree models'.
    """
    check_config(config)
    kinds = {"degree": "degree", "log_degree": "degree",
             "n_sent": "count", "n_received": "count",
             "n_currencies": "count", "n_counterparty_banks": "count",
             "active_span_days": "timing", "mean_gap_hours": "timing"}
    rows = []
    for column in FEATURE_CONFIGS[config]:
        rows.append({
            "feature": column,
            "group": "topology" if column in TOPOLOGY_FEATURES else "attributes",
            "group_name": ("Graph topology (config A)" if column in TOPOLOGY_FEATURES
                           else "Transaction attributes (config B)"),
            "kind": kinds.get(column, "amount"),
        })
    return pd.DataFrame(rows, columns=["feature", "group", "group_name", "kind"])


def build_graph_data(dataset, config="topology", results_dir="results",
                     backend="pandas", cache=True, edge_dtype=torch.int64):
    """The PyG ``Data`` object for ``dataset`` under feature ``config``.

    Returns ``(data, node_order, seconds)``, where ``seconds`` is the
    preprocessing time the scalability table reports against GARG-AML's
    Louvain + edge-removal step.

    ``edge_dtype`` exists for LI-Large, where ``torch.int32`` would halve the
    5.6 GB ``edge_index``. **Verified not to work on torch 2.3.1 /
    torch-geometric 2.5.3**: the forward pass dies in ``scatter()`` with
    "Expected dtype int64 for index". The parameter is kept because a newer
    build may lift that, but the task-1 note suggesting int32 as an LI-Large
    memory saving does not hold today -- plan for the full int64 footprint
    and the >=64 GB host RAM that implies, and re-test before relying on it.
    """
    check_config(config)
    started = time.perf_counter()

    edge_index, node_order, degree = build_graph_structure(
        dataset, results_dir=results_dir, backend=backend, cache=cache)
    x = build_features(dataset, node_order, degree, config,
                       results_dir=results_dir, cache=cache)

    data = Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=edge_dtype))

    return data, node_order, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphSAGEModel(torch.nn.Module):
    """Plain 2-layer GraphSAGE -- the same receptive field as GARG-AML's
    second-order neighbourhood, which is the whole point of the comparison.
    Outputs one raw logit per node (BCEWithLogitsLoss, not a softmax)."""

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


def peak_host_memory_mb():
    """Peak resident set size of this process, in MB.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux -- the runs that
    matter happen on Linux (VSC) and the development happens on macOS, so
    getting this wrong would misreport by 1024x on exactly one of them.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 ** 2) if sys.platform == "darwin" else usage / 1024


def peak_gpu_memory_mb(device):
    """Peak GPU allocation in MB; NaN where the backend cannot report one."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")  # MPS exposes current, not peak, allocation


def reset_peak_gpu_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def validation_split(y, train_mask, val_fraction=0.1, seed=SEED):
    """Carve a stratified validation slice out of the training fold.

    The test fold is never touched: early stopping must not see it, or the
    "same folds as the tree models" claim would be comparing a model that
    peeked against models that did not. Returns ``(fit_mask, val_mask)``, or
    ``(train_mask, None)`` when the fold has too few positives to stratify --
    in which case the caller trains for a fixed number of epochs and says so
    in the run status rather than silently early-stopping on nothing.
    """
    train_idx = np.flatnonzero(train_mask)
    y_train = y[train_idx]

    # Enough positives that the validation slice is expected to contain at
    # least one. A stratified split of, say, 3 positives at 10% can hand back
    # a validation set with none, and average_precision_score on an
    # all-negative slice returns 0.0 every epoch -- early stopping would then
    # be driven by a constant, which is worse than not early stopping at all.
    # Expect this to bite at cut-off 0.9, exactly where the tree models fail.
    min_positives = max(2, int(np.ceil(1 / val_fraction)))
    if int(y_train.sum()) < min_positives or int((1 - y_train).sum()) < min_positives:
        return train_mask, None

    fit_idx, val_idx = train_test_split(
        train_idx, test_size=val_fraction, stratify=y_train, random_state=seed)

    fit_mask = np.zeros_like(train_mask)
    val_mask = np.zeros_like(train_mask)
    fit_mask[fit_idx] = True
    val_mask[val_idx] = True
    return fit_mask, val_mask


def train_fold(data, train_mask, y, device, epochs=50, patience=5, hidden_channels=64,
               dropout=0.2, lr=1e-3, batch_size=1024, num_neighbors=(25, 10),
               val_fraction=0.1, seed=SEED, num_workers=0, checkpoint_path=None,
               resume=True, verbose=True):
    """Train one GraphSAGE model on ``train_mask``, transductively.

    The loader samples neighbourhoods from the *whole* graph -- message
    passing sees every node, train and test alike, because this pipeline is
    transductive throughout (see CLAUDE.md). Only the seed nodes, and
    therefore the loss, are restricted to the training fold.

    Early stopping is on validation **AUC-PR**, not loss: AUC-PR is the
    paper's primary threshold-free metric, and at a 0.1% positive rate the
    loss can improve for a long time while the ranking does not. The best
    state by validation AUC-PR is restored before returning, so the returned
    model is the early-stopped one rather than the last epoch's.

    ``checkpoint_path`` writes optimiser and model state every epoch and
    resumes from it, so a cluster wall-clock timeout costs one epoch rather
    than the whole run.

    Returns ``(model, info)`` with the timing and stopping diagnostics the
    scalability table needs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    fit_mask, val_mask = validation_split(y, train_mask, val_fraction, seed)

    # Reassigned in place each call rather than cloning: edge_index alone is
    # twice HI-Small's ~5M edges, and training here is sequential.
    data.y = torch.tensor(y, dtype=torch.float)

    loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=torch.tensor(fit_mask), shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        filter_per_worker=num_workers > 0,
    )

    model = GraphSAGEModel(data.num_node_features, hidden_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = int(y[fit_mask].sum())
    n_neg = int(fit_mask.sum()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # What a checkpoint must agree with to be resumable. Without this, a
    # checkpoint left behind by a short test run (or by a different
    # hyperparameter setting) is silently resumed by the next real run, which
    # then reports epochs it never trained under the settings it claims.
    signature = {"in_channels": data.num_node_features, "hidden_channels": hidden_channels,
                 "dropout": dropout, "lr": lr, "batch_size": batch_size,
                 "num_neighbors": tuple(num_neighbors), "seed": seed}

    start_epoch, best_ap, best_epoch, best_state, stale = 0, -np.inf, -1, None, 0

    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if checkpoint.get("signature") != signature:
            if verbose:
                print("    ignoring "+checkpoint_path+": trained under different "
                      "settings, starting fresh")
        else:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_ap, best_epoch = checkpoint["best_ap"], checkpoint["best_epoch"]
            best_state, stale = checkpoint["best_state"], checkpoint["stale"]
            if verbose:
                print("    resumed from "+checkpoint_path+" at epoch "+str(start_epoch))

    started = time.perf_counter()
    epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = criterion(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * batch.batch_size
        mean_loss = total_loss / max(int(fit_mask.sum()), 1)

        if val_mask is None:
            if verbose:
                print("    epoch "+str(epoch)+": loss "+format(mean_loss, ".4f")+" (no validation slice)")
        else:
            scores = score_all(model, data, device)
            val_ap = average_precision_score(y[val_mask], scores[val_mask])
            improved = val_ap > best_ap
            if improved:
                best_ap, best_epoch, stale = val_ap, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if verbose:
                print("    epoch "+str(epoch)+": loss "+format(mean_loss, ".4f")
                      +"  val AUC-PR "+format(val_ap, ".4f")+("  *" if improved else ""))

        if checkpoint_path:
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_ap": best_ap, "best_epoch": best_epoch,
                        "best_state": best_state, "stale": stale,
                        "signature": signature}, checkpoint_path)

        if val_mask is not None and stale >= patience:
            if verbose:
                print("    early stop: no val AUC-PR improvement for "+str(patience)+" epochs")
            break

    fit_seconds = time.perf_counter() - started

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "fit_seconds": fit_seconds,
        "epochs_run": epoch + 1,
        "epochs_to_best": best_epoch + 1 if best_epoch >= 0 else np.nan,
        "best_val_AP": best_ap if np.isfinite(best_ap) else np.nan,
        "early_stopped": bool(val_mask is not None and stale >= patience),
        "validated": val_mask is not None,
    }
    return model, info


def score_all(model, data, device, batch_size=None, num_neighbors=(25, 10),
              num_workers=0):
    """Score every node.

    Default is one full-graph forward pass, which is the "inference" cost the
    scalability table reports. ``batch_size`` switches to sampled inference
    through a NeighborLoader instead -- necessary where the full graph does
    not fit on the device (LI-Large), and approximate for the same reason
    training is: the neighbourhood is sampled, not complete.
    """
    model.eval()

    if batch_size is None:
        with torch.no_grad():
            logits = model(data.x.to(device), data.edge_index.to(device))
        return torch.sigmoid(logits).cpu().numpy()

    loader = NeighborLoader(
        data, num_neighbors=list(num_neighbors), batch_size=batch_size,
        input_nodes=None, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0, filter_per_worker=num_workers > 0,
    )
    scores = np.empty(data.num_nodes, dtype=np.float32)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)[:batch.batch_size]
            seeds = batch.n_id[:batch.batch_size].cpu().numpy()
            scores[seeds] = torch.sigmoid(logits).cpu().numpy()
    return scores


def timed_score_all(model, data, device, **kwargs):
    """:func:`score_all` plus the inference time, for the timing table."""
    started = time.perf_counter()
    scores = score_all(model, data, device, **kwargs)
    return scores, time.perf_counter() - started
