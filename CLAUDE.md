# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

**GARG-AML** (*Graph-Aided Risk Guarding for Anti-Money Laundering*) is a research
codebase for detecting **smurfing** patterns in financial transaction networks. It
accompanies:

> Deprez, Baesens, Verdonck, Verbeke. *GARG-AML against Smurfing: A Scalable and
> Interpretable Graph-Based Framework for Anti-Money Laundering.* arXiv:2506.04292 (2025).
> Under revision at The Journal of Finance and Data Science (JFDS-D-26-00208).

---

## 1. Read this first — current focus

The repository is in a **peer-review revision cycle**. Two reference documents in the
project define the work; **read them before starting any task**:

- `GARG-AML_review_feedback.md` — the reviewer comments and what each one asks for.
- `GARG-AML_code_changes.md` — the 13 numbered code tasks, with full specifications.

**Task numbering.** "Task *n*" always means the numbering in `GARG-AML_code_changes.md`
— that is the canonical list. The priority table in `GARG-AML_review_feedback.md` is a
*separate* reviewer-facing ordering and is labelled **P1–P13** to avoid collision; do not
mix the two indices (code task 12 = model naming, P12 = the typo lists).

**Branch:** work on the revision branch (`revision-jfds`), never on `main` (protected).

**Task ordering is not free.** Three tasks have hard dependencies:

1. **Task 2 first** — the ranking-metrics module (Precision@K, Recall@K, AP, lift,
   TP@top-N). Not started in code, but **fully specified**: see "Design (agreed)" under
   §2 of `GARG-AML_code_changes.md` for the settled API, output schema and retrofit list.
   Implement that, do not redesign it. The module goes in **`src/utils/evaluation.py`**
   (beside `naming.py`); every script currently defines its own `evaluate_model` and all
   six must route through it. Do not let a new model build its own parallel evaluation
   path — note that `scripts/gargaml_tree_blocks.py` already did, and folding it back in
   is part of the task. Two invariants worth repeating here: `evaluate_model` returns a
   **dict**, and the shared writer must keep emitting today's
   `<dataset>_<metric>_<model>_<direction>_combined.csv` files unchanged so the
   visualisation notebooks need no edits.
2. **Task 3 second** — refactor the feature matrix into separable column groups
   (GARG-AML scores / block densities+sizes / neighbourhood summary stats). This
   unblocks the topology-only ablation and the feature-matrix documentation, and must
   land before any tree/boost re-runs. **Partly done:**
   `scripts/gargaml_tree_blocks.py` runs a block-only (group b) ablation and emits a
   feature-schema CSV, but it duplicates the whole train/eval path instead of using
   shared column groups, and the group (c) ablation the reviewers asked for is still
   missing.
3. **Task 1 third** — the GraphSAGE baseline (`pytorch-geometric`). Not started; no
   torch / torch-geometric in `requirements.txt` yet. Highest compute risk; get one
   HI-Small run working end to end before queuing LI-Large on VSC.

Remaining tasks (4–13) are largely independent loops over the existing pipeline. Two of
them already have work on disk:

- **Task 4** — `notebooks/LouvainEdgeSeverance.ipynb` quantifies the % of edges severed
  on every dataset, but only at `resolution = 10`. The resolution sweep, the
  split-pattern diagnostic and the no-Louvain run are still open.
- **Task 12** — `src/utils/naming.py` is the canonical name map and is used by
  `DistributionScores`, `VisualisationResults` and `VisualisationRunTime`. Still open:
  the `scripts/` do not route through it, and it has no entries for the block-only
  ablation or GraphSAGE.

**Before committing:** strip Jupyter notebook outputs (`nbstripout` or manual). Output
cells previously caused an HTTP 400 push failure on this repo. This is *not* automated
today — there is no `.gitattributes` and no `filter.nbstripout.clean` git config — and
four tracked notebooks (`VisualisationResults`, `LouvainEdgeSeverance`, `toyexample`,
`VisualisationRunTime`) still carry outputs. Strip before touching them.

### Settings that are becoming parameters

Several values below are hard-coded today and are *being made configurable* by the
revision. Do not add new hard-coded uses of them:

| Value | Status |
|---|---|
| Louvain `resolution = 10` | Already a keyword argument (`graph_community(G, resolution=10)`); every call site takes the default and `LouvainEdgeSeverance.ipynb` hardcodes 10. Task 4 sweeps it — thread the value through the call sites, do not add new literals |
| 70/30 single split | Task 7 adds 5 repeated seeds — split logic must accept a seed |
| Label cut-off list | Task 2 adds threshold-free ranking metrics alongside |
| Model names (`gargaml tree undirected` vs `GARG-AML Undir. Tree`) | Task 12 standardises — `src/utils/naming.py` is the one canonical scheme; route new labels through `pretty()` |

### Do not "fix" these

- The `try/except` around model training in `gargaml_tree.py` fills metric matrices
  with `NaN` when a (cutoff, pattern) cell has too few positives. That is **expected
  behaviour**, and the revision requires those gaps to be *reported*, not hidden.
- The GNN baseline must **never** receive GARG-AML scores, block densities, or block
  sizes as features. That would destroy the ablation's meaning.

### Known defect — fix as part of task 2

`evaluate_model` in `gargaml_tree.py`, `gargaml_tree_blocks.py` and all three
`gargaml_tree_synthetic*.py` computes AUC-ROC and AUC-PR from `clf.predict()` (hard 0/1
labels) rather than `predict_proba`. AUC-PR is the paper's primary threshold-free metric,
and the ranking metrics of task 2 (P@K, lift, TP@top-N) all need a continuous score, so
the new evaluation module must take scores, not labels.

Still live — all five call sites are unfixed. The fix lands in one place,
`src/utils/evaluation.py::model_scores` (`predict_proba[:, 1]`, falling back to
`decision_function`, then `-score_samples` for the isolation forest). Note that this also
*delivers* one of the metrics the reviewers ask for rather than adding it:
`average_precision_score` already **is** average precision, so it is the existing
`AUC_PR` column computed correctly, not a new column.

---

## 2. How to run things

**Always run scripts from the repository root.** Every script starts with
`os.chdir("./"); sys.path.append("./")` and uses root-relative paths
(`data/HI-Small_Trans.csv`, `results/...`). Running from inside `scripts/` breaks them.

```bash
# from the repo root, with the venv active:
python scripts/gargaml_directed.py      # step 1: compute block measures -> results/
python scripts/gargaml_tree.py          # step 2: train/evaluate       -> results/
```

Requirements: **networkx >= 3.0** (`nx.community.louvain_communities`). Measure scripts
parallelise via `multiprocessing.Pool`, capped at `min(4, cpu_count() // 2)` workers.

**Reproducibility:** Louvain uses `seed=1997`; sklearn splits/models use
`random_state=1997`. Keep these fixed when comparing runs. Where task 7 introduces
multiple seeds, 1997 stays the first seed.

---

## 3. The core idea — read before touching `src/methods/`

In a *pure smurfing* pattern, money moves from one source account to one target account
through several intermediate "mules", so source and target never transact directly. If
you order the nodes of a node's **second-order ego graph** as
`[node + 2nd-order neighbours | 1st-order neighbours]`, the adjacency matrix splits into
blocks where the **on-diagonal blocks are empty** and the **off-diagonal blocks are
dense**. GARG-AML scores each node by exactly this contrast. High score ⇒ smurfing-like
structure.

**Undirected** (paper §3.2). Order as `[v, N₂(v) | N₁(v)]`, take 3 blocks: `block1`
(upper-left, among v + 2nd-order), `block2` (upper-right, v+2nd-order vs 1st-order),
`block3` (lower-right, among 1st-order). Each block's density over its *free* entries
gives `measure_1/2/3`. Final score (Eq. 8):

```
score2 − (l1·score1 + l3·score3) / (l1 + l3)
```

with block sizes `l1 = m²−3m+2`, `l3 = n²−n`, where `n = |N₁(v)|` and `m−1 = |N₂(v)|`.
This size weighting is the `"weighted_average"` score type. Range **[−1, 1]**.

**Directed** (paper §3.3). Uni-directional flow is definitional for smurfing, so
2nd-order neighbours split into **level 0 = senders** and **level 2 = receivers**
(level 1 = the mules / 1st-order neighbours), decided via the *strong* (directed-path)
vs *weak* (undirected) 2nd-order neighbourhood, repeated on the reversed graph. The 3×3
block grid yields `measure_00` … `measure_22`. Final score (Eq. 14):

```
mean(score01, score12) − mean(score00, score02, score10, score11, score20, score21, score22)
```

`GARG_AML_node_directed` computes this for the graph and its transpose and keeps the
**max**, catching reverse-flow orientation. Range **[−1, 1]**.

> Reviewer note (see `GARG-AML_review_feedback.md`, R2 major 2): the undirected score
> outperforms the directed one, and task 6 investigates why — either benign
> bidirectional edges are over-penalised, or the level-assignment rule (Eq. 11) is too
> strict. Expect to instrument this code.

Runnable scripts persist all raw block measures + block sizes;
`src/methods/gargaml_scores.py` converts them into the final score
(`score_type="weighted_average"` is what the experiments use).

---

## 4. Pipeline

```
transaction CSV  ──construct_*_graph──▶  NetworkX (Di)Graph
        │
        ▼  graph_community()   Louvain, keep intra-community edges only
   reduced graph
        │
        ▼  GARG_AML_node_*_measures()   per node, multiprocessing.Pool
   results/<dataset>_GARGAML_<directed|undirected>.csv     (block measures + sizes)
        │
        ▼  define_gargaml_scores() + summarise_gargaml_scores() + neighbour stats
   feature table
        │
        ▼  decision tree / gradient boosting / isolation forest, evaluated vs labels
   results/<dataset>_<metric>_<model>_combined.csv  +  plots
```

The two stages are **decoupled**: `gargaml_directed.py` / `gargaml_undirected.py` write
block measures to `results/*.csv`; `gargaml_tree.py` / `gargaml_IF.py` read them back to
train and evaluate. **If you change the measure columns, both ends must be updated
together** — this is the most common way to break the pipeline, and task 3's refactor
touches exactly this boundary.

---

## 5. Repository layout

```
src/
  data/
    graph_construction.py     # CSV -> NetworkX graph
    pattern_construction.py   # parse *_Patterns.txt -> per-node AML labels
    synthetic_smurfing.py     # generate synthetic graphs (igraph) + injected patterns
    dataprep_vsc.py           # split/recombine the huge LI-Large CSV
  methods/
    GARGAML.py                # core: per-node block measures + score (entry: GARG_AML)
    gargaml_scores.py         # block measures -> "basic" / "weighted_average" scores
    utils/
      measure_functions_undirected.py   # the 3 undirected block-density measures
      measure_functions_directed.py     # the 9 directed block-density measures
      neighbourhood_functions.py        # node ordering, neighbour stats, final DataFrame
  utils/
    graph_processing.py       # graph_community() (Louvain filter), graph_degree() (hub removal)
    naming.py                 # canonical model names (task 12): MODEL_DISPLAY_NAMES, pretty()

scripts/                      # runnable entry points — run from repo ROOT
  gargaml_directed.py         # directed measures on IBM data
  gargaml_undirected.py       # undirected variant
  gargaml_directed_synth.py   # directed measures on the synthetic grid
  gargaml_undirected_synth.py # undirected synthetic variant
  gargaml_tree.py             # decision tree + boosting on IBM scores
  gargaml_tree_blocks.py      # block-only ablation (partial task 3) — duplicated eval path
  gargaml_tree_synthetic.py   # tree models on synthetic data
  gargaml_tree_synthetic_3.py # …with 3 injected patterns
  gargaml_tree_synthetic_5.py # …with 5 injected patterns
  gargaml_IF.py               # isolation forest (unsupervised)
  gargaml_link_label.py       # edge-/link-level labelling
  distribution_scores.py      # score-distribution analysis/plots
  test_parallel.py            # multiprocessing sanity check, not part of the pipeline

notebooks/                    # exploratory analysis & paper figures (run from repo root)
                              # only 6 are tracked; the exploratory ones are gitignored
                              # by name — see .gitignore
requirements.txt              # networkx>=3.0, pandas, numpy, scikit-learn, matplotlib,
                              # tqdm, igraph. No torch/torch-geometric yet (task 1)
Figure A1–A6.pdf              # appendix figures, committed at repo root
data/                         # NOT in git — place datasets here yourself
results*/, res/               # NOT in git — generated outputs
config/, lib/                 # legacy/empty, not used by the pipeline
assets/img/SmurfingNetwork.png
initial_code.py               # early reference implementation, NOT maintained — ignore
                              # (also gitignored)
```

New revision code (task 1) should follow the existing split: a module under `src/` and a
runnable entry point under `scripts/`, writing to `results/`. Do not put training logic
in notebooks.

---

## 6. Data setup (data is not committed)

`data/` is git-ignored. To reproduce results, provide:

- **IBM AMLworld** — from Kaggle
  (`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`):
  `data/HI-Small_Trans.csv`, `data/HI-Small_Patterns.txt`,
  `data/LI-Large_Trans.csv`, `data/LI-Large_Patterns.txt`.
  `LI-Large_Trans.csv` is ~16 GB — use `src/data/dataprep_vsc.py` to split into
  `LI-Large_Trans_0.csv` … `_29.csv` for chunked processing.
- **Synthetic** — generated by `src/data/synthetic_smurfing.py`. The `*_synth.py`
  scripts expect `data/edge_data_synthetic_<...>.csv` (columns `source`, `target`) and
  `data/label_data_synthetic_<...>.csv`.

Expected columns:
- IBM transactions: `Account`, `Account.1`, `From Bank`, `To Bank`, `Amount Received`,
  `Amount Paid`, `Is Laundering`.
- Synthetic edges: `source`, `target`.

**AML pattern types** (used throughout, from `*_Patterns.txt`): `FAN-OUT`, `FAN-IN`,
`GATHER-SCATTER`, `SCATTER-GATHER`, `CYCLE`, `RANDOM`, `BIPARTITE`, `STACK`, plus a
"not classified" bucket and the overall `Is Laundering` flag. GARG-AML targets
`GATHER-SCATTER` / `SCATTER-GATHER` specifically.

---

## 7. Published experimental setup (baseline to match)

These are the settings the current paper reports. Match them when reproducing; the
revision tasks extend rather than replace them.

**Datasets**
- `HI-Small`: 515,080 nodes / 5,078,345 edges / 0.1 % laundering.
- `LI-Large`: 2,054,390 nodes / 176,066,557 edges / 0.05 % laundering.
- Synthetic: 66 datasets = {Barabási-Albert, Erdős-Rényi, Watts-Strogatz} × sizes
  {100; 10,000; 100,000} × edge/rewiring params × {3, 5} injected patterns. Grid lives
  in `gargaml_directed_synth.py::construct_datasets`. Each pattern has 2–10 smurfs
  (random). Injection types: `separate` (disconnected), `new_mules` (external smurfs —
  mules added between two existing accounts), `existing_mules` (internal smurfs —
  existing nodes reused as sender/receiver/mule).

**Label construction (edge → node).** Per account, propensity = laundering-labelled
transactions / total transactions. Positive if propensity exceeds a cut-off; scripts
sweep `[0.1, 0.2, 0.3, 0.5, 0.9]`, paper headlines 0.1 / 0.5 / 0.9. Imbalance is extreme
(often <0.1 % positives).

**Models**
- Pre-processing: Louvain (NetworkX, resolution 10, seed 1997), intra-community edges
  only (Algorithm 1).
- Extension features: base GARG-AML score + neighbourhood summary stats (min/mean/max/std
  of neighbours' scores *and* degrees). Four extended models = {undirected, directed} ×
  {decision tree, gradient boosting}.
  - Decision tree: Gini, `min_samples_leaf=10`.
  - Gradient boosting: 100 trees, `max_depth=3`, `learning_rate=0.1`,
    `min_samples_leaf=10`, `random_state=1997`.
  - 70/30 stratified split on the **feature table**, not the graph (this is transductive
    — the revision requires it to be stated explicitly).
- Baselines: FlowScope and AutoAudit (default hyperparameters; AutoAudit averaged over 3
  random index initialisations). Their code lives in **separate forked repos** linked
  from the README — only the code that *analyses their output* is here. Task 1 adds
  GraphSAGE, which does live in this repo.
- Metrics: precision, F1, AUC-ROC, AUC-PR, plus average rank across the 66 synthetic
  datasets via Friedman + post-hoc Nemenyi (k=8, N=66, α=0.05), shown as
  critical-difference diagrams. Task 2 adds ranking metrics; task 8 adds the χ²/p-value
  reporting the reviewers asked for.
- Compute envelope for the largest runs: 16 h VSC budget, 200 GB.

**Headline finding:** undirected GARG-AML alone is strong; adding tree/boosting matches
or beats FlowScope/AutoAudit with far higher precision, while scaling to graphs where
AutoAudit exhausts memory/time.

> Claim caution (R2 major 5): "state-of-the-art performance", "significantly reducing
> false positives", and "first analysis of scalability" are all flagged as overclaims.
> Do not reproduce that phrasing in docstrings, READMEs, or plot titles.

---

## 8. Version control

Committed: `src/`, `scripts/`, README/LICENSE, paper figures (`Figure*.pdf`, currently
`FigureA1`–`FigureA6`), the smurfing diagram, and the six notebooks that produce paper
figures/analysis (`toyexample`, `VisualisationResults`, `VisualisationRunTime`,
`VisualisationNetwork`, `DistributionScores`, `LouvainEdgeSeverance`). Ignored: all of
`data/`, `results*/` and `res/`, trained models (`*.pkl`), GraphViz exports (`*.dot`),
scratch plots, caches, `.DS_Store`, `initial_code.py`, the exploratory notebooks
(`AnalysisData`, `AnalysisParallel`, `AnalysisSyntheticData`, `AnomalousPatterns`,
`TestCode`, `Transfer`), and the reviewer/revision notes
(`GARG-AML_review_feedback.md`, `GARG-AML_code_changes.md`) — these are internal and
must not be pushed. Rationale is inline in `.gitignore`.

The README currently lacks a reproduction map, and the manuscript claims "Code ✓" without
a repo link (R2 minor 4, task 13) — when adding new experiments, document in the README
which script reproduces which table or figure.
