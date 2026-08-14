# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this project is

**GARG-AML** (*Graph-Aided Risk Guarding for Anti-Money Laundering*) is a
research codebase for detecting **smurfing** patterns in financial transaction
networks. It accompanies the paper:

> Deprez, Baesens, Verdonck, Verbeke. *GARG-AML against Smurfing: A Scalable and
> Interpretable Graph-Based Framework for Anti-Money Laundering.* arXiv:2506.04292 (2025).

### The core idea (read this before touching `src/methods/`)

In a *pure smurfing* pattern, money moves from one source account to one target
account through several intermediate "mules", so the source and target never
transact directly. If you order the nodes of a node's **second-order ego graph**
as `[node + 2nd-order neighbours | 1st-order neighbours]`, the adjacency matrix
splits into blocks where the **on-diagonal blocks are empty (0)** and the
**off-diagonal blocks are dense (1)**. GARG-AML scores each node by exactly this
contrast in block density — a high score means the node sits in a smurfing-like
structure.

- **Undirected** (paper §3.2): order nodes as `[v, N₂(v) | N₁(v)]` and take 3
  blocks — `block1` (upper-left, among v + 2nd-order), `block2` (upper-right,
  v+2nd-order vs 1st-order), `block3` (lower-right, among 1st-order). Each block's
  *density over its free entries* gives `score1/score2/score3`
  (`measure_1/2/3` in code). Final score (Eq. 8):
  `score2 - (l1·score1 + l3·score3)/(l1+l3)`, where `l1 = m²−3m+2`, `l3 = n²−n`
  are the block sizes (so it's a **size-weighted** average, the `"weighted_average"`
  score type). `n = |N₁(v)|`, `m−1 = |N₂(v)|`. Range **[−1, 1]**; ≈1 ⇒ smurfing-like.
- **Directed** (paper §3.3): a defining trait of smurfing is *uni-directional*
  flow, so 2nd-order neighbours are split into **level 0 = senders** and
  **level 2 = receivers** (level 1 = the mules / 1st-order neighbours). This is
  decided using the *strong* (directed-path) vs *weak* (undirected) 2nd-order
  neighbourhood, repeated on the reversed graph. The 3×3 block grid gives 9
  measures (`measure_00` … `measure_22`). Final score (Eq. 14):
  `mean(score01, score12) − mean(score00,02,10,11,20,21,22)`. The code
  (`GARG_AML_node_directed`) computes this for both the graph and its transpose
  and keeps the **max**, to catch reverse-flow orientation. Range **[−1, 1]**.

Note: the runnable scripts persist all 9 (or 3) raw block measures + block sizes;
`src/methods/gargaml_scores.py` turns those into the final score above
(`score_type="weighted_average"` is what the experiments use).

The raw scores are then **aggregated with neighbourhood summary statistics**
(min/max/mean/std of neighbours' scores and degrees) and fed to a simple,
interpretable classifier (decision tree / gradient boosting / isolation forest).

## Pipeline / data flow

```
transaction CSV  ──construct_*_graph──▶  NetworkX (Di)Graph
        │
        ▼  graph_community()  (Louvain, resolution=10, keep intra-community edges only)
   reduced graph
        │
        ▼  GARG_AML_node_*_measures()  (per node, parallelised with multiprocessing.Pool)
   results/<dataset>_GARGAML_<directed|undirected>.csv   (block measures + sizes)
        │
        ▼  define_gargaml_scores() + summarise_gargaml_scores() + neighbour stats
   feature table
        │
        ▼  decision tree / boosting / isolation forest, evaluated vs ground-truth patterns
   results/<dataset>_<metric>_<model>_combined.csv  +  plots
```

The two stages are **decoupled**: the `gargaml_directed.py` / `gargaml_undirected.py`
scripts compute and persist the block measures to `results/*.csv`; the
`gargaml_tree.py` / `gargaml_IF.py` scripts read those CSVs back to train and
evaluate classifiers. If you change the measure columns, both ends must agree.

## Repository layout

```
src/
  data/
    graph_construction.py     # CSV -> NetworkX graph (IBM cols Account/Account.1; synthetic cols source/target)
    pattern_construction.py   # parse *_Patterns.txt, build per-node AML labels (FAN-OUT, FAN-IN, CYCLE, ...)
    synthetic_smurfing.py     # generate synthetic graphs (igraph) + injected smurfing patterns
    dataprep_vsc.py           # split the huge LI-Large CSV into k chunks / recombine
  methods/
    GARGAML.py                # core: per-node block measures + the GARG-AML score (entry: GARG_AML)
    gargaml_scores.py         # turn block measures into "basic" / "weighted_average" scores
    utils/
      measure_functions_undirected.py   # the 3 undirected block-density measures
      measure_functions_directed.py     # the 9 directed block-density measures
      neighbourhood_functions.py        # node ordering + neighbour summary stats + assemble final DataFrame
  utils/
    graph_processing.py       # graph_community() (Louvain filter) and graph_degree() (hub removal)

scripts/                      # runnable entry points (run from the repo ROOT — see below)
  gargaml_directed.py         # compute directed measures on IBM data  -> results/*.csv
  gargaml_undirected.py       # undirected variant
  gargaml_directed_synth.py   # directed measures on the synthetic dataset grid
  gargaml_undirected_synth.py # undirected synthetic variant
  gargaml_tree.py             # train/evaluate decision tree + boosting on IBM scores
  gargaml_tree_synthetic*.py  # tree models on synthetic data (the _3 / _5 suffix = #injected patterns)
  gargaml_IF.py               # isolation-forest (unsupervised) variant
  gargaml_link_label.py       # edge-/link-level labelling
  distribution_scores.py      # score-distribution analysis/plots

notebooks/                    # exploratory analysis & paper figures (run from repo root)
data/                         # NOT in git — you must place datasets here yourself (see below)
results*/, res/               # NOT in git — generated outputs
assets/img/SmurfingNetwork.png# the smurfing diagram used in the README
```

## Running the code

**Always run scripts from the repository root.** Every script begins with
`os.chdir("./"); sys.path.append("./")` and uses paths relative to the root
(e.g. `data/HI-Small_Trans.csv`, `results/...`). Running from inside `scripts/`
will break those paths.

```bash
# from the repo root, with the venv active:
python scripts/gargaml_directed.py      # step 1: compute measures -> results/
python scripts/gargaml_tree.py          # step 2: train/evaluate -> results/
```

Most measure scripts parallelise with `multiprocessing.Pool` capped at
`min(4, cpu_count() // 2)` workers.

## Data setup (important — data is not committed)

The `data/` directory is git-ignored. To reproduce results you must provide:

- **IBM AML data** (real): download from Kaggle
  (https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
  and place the files in `data/`:
  - `data/HI-Small_Trans.csv`, `data/HI-Small_Patterns.txt`
  - `data/LI-Large_Trans.csv`, `data/LI-Large_Patterns.txt`
  - `LI-Large_Trans.csv` is ~16 GB; use `src/data/dataprep_vsc.py` to split it
    into `LI-Large_Trans_0.csv` … `_29.csv` for chunked processing.
- **Synthetic data:** generated by `src/data/synthetic_smurfing.py`. The
  `*_synth.py` scripts expect files named `data/edge_data_synthetic_<...>.csv`
  (columns `source`, `target`) and `data/label_data_synthetic_<...>.csv`.

Expected CSV columns:
- IBM transactions: `Account`, `Account.1`, `From Bank`, `To Bank`,
  `Amount Received`, `Amount Paid`, `Is Laundering`.
- Synthetic edges: `source`, `target`.

## Experimental setup (from the paper, for reproduction)

These are the exact settings the paper reports — match them when reproducing or
extending results.

**Datasets**
- **IBM open-source** (Altman et al. 2023; Egressy et al. 2024):
  - `HI-Small`: 515,080 nodes / 5,078,345 edges / **0.1 %** laundering.
  - `LI-Large`: 2,054,390 nodes / 176,066,557 edges / **0.05 %** laundering.
- **Synthetic**: 66 datasets = {Barabási-Albert, Erdős-Rényi, Watts-Strogatz} ×
  sizes {100; 10,000; 100,000} × edge/rewiring params × {3, 5} injected patterns.
  Generators and parameter grid live in `gargaml_directed_synth.py::construct_datasets`.
  Each pattern has 2–10 smurfs (random). Three **injection types**:
  `separate` (disconnected), `new_mules` = "external smurfs" (mules added between
  two existing accounts), `existing_mules` = "internal smurfs" (existing nodes
  reused as sender/receiver/mule).

**Label construction** (edge labels → node labels)
- The IBM data labels *transactions* with 8 structural pattern types
  (`FAN-OUT, FAN-IN, GATHER-SCATTER, SCATTER-GATHER, CYCLE, RANDOM, BIPARTITE,
  STACK`) plus a "not classified" bucket and the overall `Is Laundering` flag.
- Per account, a **propensity score** = (laundering-labelled txns) / (total txns).
- A node is labelled positive if that ratio exceeds a **cut-off**; the scripts
  sweep `[0.1, 0.2, 0.3, 0.5, 0.9]` (the paper headlines 0.1 / 0.5 / 0.9).
  Class imbalance is extreme — often <0.1 % positives; the `try/except` NaN-fill
  in `gargaml_tree.py` handles cells with too few positives to train.

**Models & evaluation**
- Pre-processing: Louvain communities (NetworkX, **resolution = 10**, seed 1997),
  keep intra-community edges only (Algorithm 1).
- Extension features: base GARG-AML score **+ neighbourhood summary stats**
  (min/mean/max/std of neighbours' scores *and* degrees). Four extended models =
  {undirected, directed} × {decision tree, gradient boosting}.
  - Decision tree: Gini, `min_samples_leaf=10`.
  - Gradient boosting: 100 trees, `max_depth=3`, `learning_rate=0.1`,
    `min_samples_leaf=10` (these are the sklearn defaults the code relies on,
    plus `random_state=1997`).
  - **70/30 stratified** train/test split on the *feature table* (not the graph).
- Baselines: **FlowScope** and **AutoAudit** (default hyperparameters; AutoAudit
  averaged over 3 random index initialisations). Their code is in separate forked
  repos, not here.
- Metrics: precision, F1, AUC-ROC, AUC-PR, and **average rank** across the 66
  synthetic datasets, tested with the Friedman + post-hoc Nemenyi tests (k=8
  methods, N=66, α=0.05), visualised as critical-difference diagrams.
- Headline finding: undirected GARG-AML alone is strong; adding tree/boosting
  matches or beats FlowScope/AutoAudit with far higher precision (>70 % of alerts
  are true positives vs <50 % for the baselines), while scaling to the largest
  graphs where AutoAudit runs out of memory/time (16 h VSC budget, 200 GB).

## Conventions & gotchas

- **networkx >= 3.0** is required (`nx.community.louvain_communities`).
- **Reproducibility:** Louvain uses `seed=1997`; sklearn splits/models use
  `random_state=1997`. Keep these fixed when comparing runs.
- **Ground-truth pattern labels** come from the `*_Patterns.txt` files and the
  `Is Laundering` flag. AML pattern types: `FAN-OUT, FAN-IN, GATHER-SCATTER,
  SCATTER-GATHER, CYCLE, RANDOM, BIPARTITE, STACK`.
- The `try/except` around model training in `gargaml_tree.py` deliberately fills
  the metric matrices with `NaN` when a (cutoff, pattern) cell has too few
  positives — that is expected, not a bug to "fix".
- **State-of-the-art baselines** (FlowScope, AutoAudit) live in separate forked
  repos (linked from the README), not here. Only the code that *analyses their
  output* is in this repo.
- `initial_code.py` at the root is an early reference implementation, not part of
  the maintained pipeline.

## What is and isn't version-controlled

Committed: source (`src/`, `scripts/`), notebooks, README/LICENSE, the paper
figures (`Figure*.pdf`), and the smurfing diagram. Excluded via `.gitignore`:
all of `data/`, the `results*/` and `res/` outputs, trained models (`*.pkl`),
GraphViz exports (`*.dot`), scratch plots, caches, and `.DS_Store`. See
`.gitignore` for the rationale inline.
