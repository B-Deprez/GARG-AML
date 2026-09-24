"""
Diagnostics for the directed GARG-AML score.

The directed variant of Section 3.3 is motivated by uni-directional flow
being definitional for smurfing, yet it scores below the undirected variant
of Section 3.2. Two mechanisms can account for that: the penalty carried by
bidirectional edges, and the level-assignment rule of Eq. 11. This module
instruments both, per node, so they can be told apart with numbers. Nothing
here changes the pipeline: :func:`diagnose_node` recomputes the score
itself and is never called from ``GARGAML.py``.

What Eq. 11 assigns
-------------------
``GARG_AML_nodeselection_directed`` splits the undirected second-order ego
graph into three levels:

* **level 1** -- every *undirected* distance-1 neighbour. Direction plays
  no part here.
* **level 0** -- the node itself, plus every distance-2 node reachable by
  a directed path of length <= 2 in *neither* direction.
* **level 2** -- every remaining distance-2 node, i.e. every one reachable
  by a directed 2-path *either* forwards or backwards.

Section 3.3 motivates the levels as **senders (0) -> mules (1) ->
receivers (2)**. The implementation does not separate senders from
receivers: a node that reaches ``v`` in two hops (a sender, which §3.3
puts at level 0) and a node ``v`` reaches in two hops (a receiver, level
2) both land in level 2. :func:`reachability_split` counts the four cases
separately, so ``reverse_only`` counts the nodes placed at level 2 where
§3.3 puts them at level 0, and ``neither`` counts the ones Eq. 11 leaves
unresolved and groups with the node itself at level 0.

The score variants
------------------
Each variant isolates one candidate cause; all five are computed on the
same ego graph so their differences are attributable.

``directed``
    Eq. 14 as it stands. The baseline.
``transpose``
    Eq. 14 with the reward blocks taken on the other side of the diagonal
    (``mean(m10, m21)`` rewarded). A node whose flow runs the other way
    scores here what a source scores under ``directed``.
``max_transpose``
    ``max(directed, transpose)``: orientation-blind. The gap to
    ``directed`` measures the cost of Eq. 14's fixed orientation. It is a
    diagnostic upper bound, not a proposed score: Eq. 14 has no such step.
``flow_split``
    Eq. 14 under the §3.3 level assignment: senders (reverse-reachable)
    move from level 2 to level 0, receivers stay at level 2. Isolates the
    level rule from the orientation of Eq. 14 itself.
``no_reciprocal``
    Eq. 14 after deleting both directions of every reciprocal edge pair in
    the ego graph, with the levels recomputed on that graph. Measures how
    much of the score two-way relationships account for.

Reciprocal edges, and why the count is exact
--------------------------------------------
Eq. 14 rewards blocks (0,1) and (1,2) and penalises the other seven. Both
reward blocks are off-diagonal, and neither of their mirror images
((1,0) and (2,1)) is a reward block. So **every reciprocal pair between
consecutive levels necessarily puts one unit of mass in a penalty block**
-- there is no orientation of the pair that avoids it.
:func:`reciprocal_census` counts those pairs and splits them by where
their two directions land.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .utils.measure_functions_directed import (
    measure_00_function, measure_01_function, measure_02_function,
    measure_10_function, measure_11_function, measure_12_function,
    measure_20_function, measure_21_function, measure_22_function,
)
from .utils.neighbourhood_functions import GARG_AML_nodeselection_undirected

# Block keys in the order Eq. 14 reads them.
MEASURE_KEYS = ["00", "01", "02", "10", "11", "12", "20", "21", "22"]

# Eq. 14: the two blocks a pure smurfing pattern makes dense, and the seven
# it makes empty.
REWARD_BLOCKS = ("01", "12")
PENALTY_BLOCKS = ("00", "02", "10", "11", "20", "21", "22")

# The mirror image of Eq. 14, used by the ``transpose`` variant.
REWARD_BLOCKS_T = ("10", "21")
PENALTY_BLOCKS_T = ("00", "01", "02", "11", "12", "20", "22")

SCORE_VARIANTS = ["directed", "transpose", "max_transpose", "flow_split",
                  "no_reciprocal"]


# ---------------------------------------------------------------------------
# Level assignment
# ---------------------------------------------------------------------------

def ego_graphs(node, G, G_und, G_rev):
    """The three ego graphs ``GARG_AML_node_directed_measures`` builds.

    Reproduced here rather than imported because the diagnosis needs the
    intermediate objects, not just the measures; the construction is
    equivalent to that function's first three lines.
    """
    G_ego_und = nx.ego_graph(G_und, node, 2)
    G_ego = nx.subgraph(G, G_ego_und.nodes)
    G_ego_rev = nx.ego_graph(G_rev, node, 2)
    return G_ego, G_ego_und, G_ego_rev


def reachability_split(node, G_ego, G_ego_und, G_ego_rev):
    """Partition the second-order ego graph by *directed* reachability.

    Returns a dict with the level-1 neighbours and the four disjoint
    classes of distance-2 node:

    ``forward_only``
        reachable from ``node`` by a directed path of length <= 2, and not
        the other way round -- a **receiver**, level 2 under both Eq. 11
        and §3.3.
    ``reverse_only``
        reaches ``node`` in <= 2 directed hops but is not reachable from
        it -- a **sender**. §3.3 puts these at level 0; Eq. 11 as
        implemented puts them at level 2.
    ``both``
        reachable in both directions, i.e. sitting on a directed cycle
        through ``node``'s neighbourhood. Ambiguous: no level assignment
        can be right about it, so it is reported separately rather than
        folded into either side.
    ``neither``
        at undirected distance 2 with no directed 2-path either way -- for
        instance two accounts paying into the same mule. **Unresolved by
        Eq. 11**, which groups them with ``node`` itself at level 0.
    """
    nodes_1 = set(G_ego_und.neighbors(node))
    nodes_1.discard(node)

    forward = set(nx.ego_graph(G_ego, node, radius=2).nodes)
    reverse = set(nx.ego_graph(G_ego_rev, node, radius=2).nodes)

    distance_2 = set(G_ego.nodes) - nodes_1 - {node}

    return {
        "node": node,
        "nodes_1": nodes_1,
        "forward_only": distance_2 & forward - reverse,
        "reverse_only": distance_2 & reverse - forward,
        "both": distance_2 & forward & reverse,
        "neither": distance_2 - forward - reverse,
    }


def levels_published(split):
    """(level 0, level 1, level 2) exactly as Eq. 11 is implemented.

    Every distance-2 node reachable in *either* direction goes to level 2;
    the unresolved ones join ``node`` at level 0.
    :func:`check_against_pipeline` asserts this matches
    ``GARG_AML_nodeselection_directed``.
    """
    nodes_0 = [split["node"]] + sorted(split["neither"], key=str)
    nodes_1 = sorted(split["nodes_1"], key=str)
    nodes_2 = sorted(split["forward_only"] | split["reverse_only"] | split["both"],
                     key=str)
    return nodes_0, nodes_1, nodes_2


def levels_flow_split(split):
    """(level 0, level 1, level 2) under §3.3's senders -> mules -> receivers.

    Senders (``reverse_only``) move to level 0, where Section 3.3 puts
    them, instead of sharing level 2 with the receivers. ``both`` stays at
    level 2 -- it is on a cycle, so it is not a sender in any useful sense
    -- and ``neither`` stays at level 0, as Eq. 11 already has it.
    """
    nodes_0 = ([split["node"]]
               + sorted(split["neither"] | split["reverse_only"], key=str))
    nodes_1 = sorted(split["nodes_1"], key=str)
    nodes_2 = sorted(split["forward_only"] | split["both"], key=str)
    return nodes_0, nodes_1, nodes_2


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def block_measures(adj_full, size_0, size_1, size_2):
    """The nine directed block densities, through the pipeline's own functions.

    Imported rather than reimplemented, so the diagnosis and the pipeline
    cannot disagree about what a block density is.
    """
    return {
        "00": measure_00_function(adj_full, size_0)[0],
        "01": measure_01_function(adj_full, size_0, size_1)[0],
        "02": measure_02_function(adj_full, size_0, size_1, size_2)[0],
        "10": measure_10_function(adj_full, size_0, size_1)[0],
        "11": measure_11_function(adj_full, size_0, size_1)[0],
        "12": measure_12_function(adj_full, size_0, size_1, size_2)[0],
        "20": measure_20_function(adj_full, size_0, size_1, size_2)[0],
        "21": measure_21_function(adj_full, size_0, size_1)[0],
        "22": measure_22_function(adj_full, size_0, size_1, size_2)[0],
    }


def prefix_block_measures(adj_full, size_0, size_1, size_2):
    """The nine block densities under an alternative empty-block convention.

    A sensitivity variant, deliberately a second implementation and used
    only by the diagnosis: it quantifies how much of the directed score
    rests on the treatment of empty blocks. Both conventions below apply
    only when ``size_2 == 0``, i.e. to nodes with no level-2 neighbours:

    * ``measure_12`` is 1 for an empty block unconditionally -- full credit
      on a **reward** block for a node with no level-2 neighbours at all.
    * ``measure_20/21/22`` slice with ``adj_full[-size_2:]``, and ``-0:``
      spans the whole array rather than an empty one, so three **penalty**
      blocks are computed from unrelated parts of the matrix.
    """
    measures = block_measures(adj_full, size_0, size_1, size_2)

    if adj_full[size_0:size_0 + size_1, size_0 + size_1:].size == 0:
        measures["12"] = 1  # an empty reward block counts as dense here

    if size_2 == 0:
        # -0: spans the whole array, which is what this variant measures.
        for key, piece in (("20", adj_full[-size_2:, :size_0]),
                           ("21", adj_full[-size_2:, size_0:size_0 + size_1]),
                           ("22", adj_full[-size_2:, -size_2:])):
            reduced = piece.size - size_2 if key in ("20", "22") else piece.size
            measures[key] = piece.sum() / reduced if reduced > 0 else 0

    return measures


def eq_14(measures, reward=REWARD_BLOCKS, penalty=PENALTY_BLOCKS):
    """Eq. 14: mean of the reward blocks minus mean of the penalty blocks."""
    return (np.mean([measures[k] for k in reward])
            - np.mean([measures[k] for k in penalty]))


def measures_for_levels(G_ego, nodes_0, nodes_1, nodes_2):
    """Block densities for one level assignment of one ego graph."""
    ordered = list(nodes_0) + list(nodes_1) + list(nodes_2)
    adj = nx.to_numpy_array(G_ego, nodelist=ordered, dtype=int)
    return block_measures(adj, len(nodes_0), len(nodes_1), len(nodes_2))


# ---------------------------------------------------------------------------
# Bidirectional edges
# ---------------------------------------------------------------------------

def reciprocal_pairs(G_ego):
    """Unordered pairs ``{u, v}`` carrying edges in both directions."""
    return {frozenset((u, v)) for u, v in G_ego.edges() if G_ego.has_edge(v, u)}


def reciprocal_census(G_ego, nodes_0, nodes_1, nodes_2):
    """How many reciprocal pairs there are, and how much penalty they carry.

    Because Eq. 14's two reward blocks are off-diagonal and neither of
    their mirror images is also a reward block, a reciprocal pair spanning
    levels 0-1 or 1-2 puts exactly one of its two directions in a reward
    block and the other in a penalty block; ``penalised`` counts those.

    Pairs inside one level, or spanning levels 0-2, land in penalty blocks
    on both sides; they are counted as ``both_penalty``. A pair cannot
    land in a reward block on both sides.
    """
    level_of = {}
    for name, group in (("0", nodes_0), ("1", nodes_1), ("2", nodes_2)):
        for n in group:
            level_of[n] = name

    pairs = reciprocal_pairs(G_ego)
    penalised = both_penalty = 0
    for pair in pairs:
        u, v = tuple(pair)
        a, b = level_of[u] + level_of[v], level_of[v] + level_of[u]
        hits = sum(block in REWARD_BLOCKS for block in (a, b))
        if hits == 1:
            penalised += 1
        else:
            both_penalty += 1

    return {
        "n_edges": G_ego.number_of_edges(),
        "n_reciprocal": len(pairs),
        "reciprocal_penalised": penalised,
        "reciprocal_both_penalty": both_penalty,
    }


def drop_reciprocal(G_ego):
    """A copy of ``G_ego`` with both directions of every reciprocal pair gone.

    Both directions, not one: the question is what the score would be if
    two-way relationships were not evidence at all, which means removing
    their reward mass along with their penalty mass. Keeping one direction
    would instead invent a flow direction the data does not support.
    """
    H = nx.DiGraph()
    H.add_nodes_from(G_ego.nodes(data=True))
    H.add_edges_from((u, v) for u, v in G_ego.edges() if not G_ego.has_edge(v, u))
    return H


# ---------------------------------------------------------------------------
# Per-node diagnosis
# ---------------------------------------------------------------------------

def diagnose_node(node, G, G_und, G_rev):
    """Every level count, reciprocal count and score variant for one node.

    Returns a flat dict, one row of the diagnosis table. ``G`` is the
    directed (Louvain-reduced, or not -- the caller decides) graph,
    ``G_und`` its undirected view and ``G_rev`` its reverse; building
    those three once in the caller is what keeps this affordable.
    """
    G_ego, G_ego_und, G_ego_rev = ego_graphs(node, G, G_und, G_rev)
    split = reachability_split(node, G_ego, G_ego_und, G_ego_rev)

    nodes_0, nodes_1, nodes_2 = levels_published(split)
    measures = measures_for_levels(G_ego, nodes_0, nodes_1, nodes_2)

    directed = eq_14(measures)
    transpose = eq_14(measures, REWARD_BLOCKS_T, PENALTY_BLOCKS_T)

    # The same score under the alternative empty-block convention.
    ordered = list(nodes_0) + list(nodes_1) + list(nodes_2)
    adj = nx.to_numpy_array(G_ego, nodelist=ordered, dtype=int)
    prefix = eq_14(prefix_block_measures(adj, len(nodes_0), len(nodes_1),
                                         len(nodes_2)))

    f0, f1, f2 = levels_flow_split(split)
    flow_split = eq_14(measures_for_levels(G_ego, f0, f1, f2))

    # Recomputed from scratch on the de-reciprocated ego graph, levels
    # included: removing edges changes directed reachability, so reusing the
    # earlier levels would mix the two effects.
    H = drop_reciprocal(G_ego)
    H_split = reachability_split(node, H, G_ego_und, H.reverse(copy=True))
    h0, h1, h2 = levels_published(H_split)
    no_reciprocal = eq_14(measures_for_levels(H, h0, h1, h2))

    # Eq. 8 on the same neighbourhood, as the reference point.
    G_ego_u2 = nx.ego_graph(G_und, node, 2)
    u1, u2, u_ordered = GARG_AML_nodeselection_undirected(G_ego_u2, node)
    adj_u = nx.to_numpy_array(G_ego_u2, nodelist=u_ordered, dtype=int)
    from .utils.measure_functions_undirected import (
        measure_1_function, measure_2_function, measure_3_function)
    p1 = [len(u2) + 1, len(u2) + 1]
    p2 = [len(u1), len(u2) + 1]
    p3 = [len(u1), len(u1)]
    m1 = measure_1_function(p1, adj_u)[0]
    m2 = measure_2_function(p1, p2, adj_u)[0]
    m3 = measure_3_function(p1, p2, p3, adj_u)[0]
    undirected = m2 - (m1 + m3) / 2

    row = {
        "node": node,
        # Level census; an empty level 2 is flagged separately, since the
        # empty-block conventions only bite there.
        "n_level0": len(nodes_0),
        "n_level1": len(nodes_1),
        "n_level2": len(nodes_2),
        "empty_level2": int(len(nodes_2) == 0),
        # Eq. 11's four distance-2 classes.
        "d2_forward_only": len(split["forward_only"]),
        "d2_reverse_only": len(split["reverse_only"]),
        "d2_both": len(split["both"]),
        "d2_neither": len(split["neither"]),
        # Scores.
        "directed": directed,
        "transpose": transpose,
        "max_transpose": max(directed, transpose),
        "flow_split": flow_split,
        "no_reciprocal": no_reciprocal,
        "undirected": undirected,
        "directed_prefix": prefix,
        # The effect sizes, so the table can be read without recomputing.
        "gain_max_transpose": max(directed, transpose) - directed,
        "gain_flow_split": flow_split - directed,
        "gain_no_reciprocal": no_reciprocal - directed,
        "gap_to_undirected": undirected - directed,
        "prefix_inflation": prefix - directed,
    }
    row.update(reciprocal_census(G_ego, nodes_0, nodes_1, nodes_2))
    row.update({"measure_" + k: measures[k] for k in MEASURE_KEYS})
    return row


def check_against_pipeline(node, G, G_und, G_rev, tol=1e-12):
    """Assert this module reproduces the pipeline's score for ``node``.

    The diagnosis is only worth reading if its ``directed`` column is the
    same number the pipeline writes, so this runs on a sample at the start
    of every diagnosis run.
    """
    from .GARGAML import GARG_AML_node_directed_measures
    from .utils.neighbourhood_functions import GARG_AML_nodeselection_directed

    G_ego, G_ego_und, G_ego_rev = ego_graphs(node, G, G_und, G_rev)
    p0, p1, p2, _ = GARG_AML_nodeselection_directed(G_ego, G_ego_und, G_ego_rev, node)

    split = reachability_split(node, G_ego, G_ego_und, G_ego_rev)
    n0, n1, n2 = levels_published(split)
    if (set(p0), set(p1), set(p2)) != (set(n0), set(n1), set(n2)):
        raise AssertionError(
            f"level assignment differs from the pipeline at node {node!r}: "
            f"pipeline {(len(p0), len(p1), len(p2))} vs diagnosis "
            f"{(len(n0), len(n1), len(n2))}")

    pipeline = dict(zip(MEASURE_KEYS,
                        GARG_AML_node_directed_measures(node, G, G_und, G_rev)))
    mine = measures_for_levels(G_ego, n0, n1, n2)
    for key in MEASURE_KEYS:
        if abs(pipeline[key] - mine[key]) > tol:
            raise AssertionError(
                f"measure_{key} differs at node {node!r}: "
                f"pipeline {pipeline[key]} vs diagnosis {mine[key]}")
    return True
