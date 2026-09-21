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

**Task ordering is not free.** Three tasks have hard dependencies. All three are now
code-complete; the chain still governs **re-runs** — a tree/boost re-run picks up
task 3's configs and reports through task 2's module, and GraphSAGE consumes task 7's
folds:

1. **Task 2 — DONE.** The ranking-metrics module is **`src/utils/evaluation.py`**
   (beside `naming.py`): Precision@K, Recall@K, lift@K, TP@K and a tie diagnostic at
   `ALERT_SIZES = [50, 100, 500, 1000]`, beside the legacy precision / F1 / AUC-ROC /
   AUC-PR. No script defines its own `evaluate_model` any more — all six route through
   it (commit `cf07af0` plus the five retrofits), including `gargaml_tree_blocks.py`,
   whose parallel evaluation path was folded back in. Do not let a new model build
   another one. Four facts to carry forward:
   - `evaluate_model` returns a **dict**, not the 4-tuple the scripts used to unpack.
   - The writer still emits today's `<dataset>_<metric>_<model>_<direction>_combined.csv`
     matrices unchanged, so the visualisation notebooks needed no edits.
   - Those matrices carry the **legacy four metrics only** (`LEGACY_METRICS`). The
     ranking metrics live in the tidy `<dataset>_<direction><suffix>_metrics.csv` that
     `write_metrics` writes beside them, long-form with a separate `K` column.
   - `LEGACY_MODEL_TOKENS` is a deliberate allow-list: a model key with no entry raises
     rather than writing an orphan CSV no notebook reads, so **add a token when you add
     a model**.

   **What is outstanding is the reporting, not the code.** The tree grid's ranking
   metrics on disk come from the 15 Sep single-split (70/30) run — the tidy files have
   no `fold` column — so they predate task 7's CV and have to be regenerated before they
   can go in the paper. P2 is answered in code and unanswered on paper until then.
2. **Task 3 — DONE.** The feature matrix is split into four separable column groups in
   **`src/utils/features.py`** (a: score, b: block densities+sizes, c: own+neighbour
   degree stats, d: neighbour score stats) and four named configs: `full` (published),
   `blocks`, `topology` (the (c)-only ablation R2-M5 asked for) and `all`.
   `scripts/gargaml_tree.py` runs all four over both directions off one data preparation;
   `gargaml_tree_blocks.py` is now a thin entry point on that shared path. Two facts worth
   carrying forward: **`full` is (a)+(c)+(d), never (b)** — the published model never
   received block densities, so that confound was never in the published numbers — and
   **`topology` is direction-free**, because group (c) is computed on the undirected
   reduced graph either way, so it runs once (`is_direction_free()`), not once per
   direction.
3. **Task 1 third — code DONE, runs outstanding.** The GraphSAGE baseline lives in
   `src/methods/graphsage.py` + `scripts/graphsage_baseline.py`, with both feature
   configs (`topology` = degree/log-degree strict parity, `attributes` = + amount,
   count, currency, bank and timing aggregates), early stopping on validation AUC-PR
   off a 10% slice carved from the *training* fold, per-epoch checkpointing, and
   separate preprocessing / fit / inference timing plus peak host and GPU memory.
   `torch` / `torch-geometric` are in `requirements.txt`. **Never** feed it GARG-AML
   scores, block densities or block sizes — that rule holds by construction here:
   both configs are built from the raw transaction file, never from
   `results/*_GARGAML_*.csv`.

   Two facts to carry forward. The graph structure is built **without NetworkX** by
   default (`backend="pandas"` in `build_graph_structure`) — verified to produce an
   identical node and edge set to `construct_IBM_graph` on HI-Small, at ~3.5x the
   speed, and it is what makes LI-Large's 176M edges plausible. And the sweep is a
   **logged reduction**, not the full grid: 3 cut-offs x 3 targets x 5 folds x 2
   configs = 90 fits on HI-Small, matching task 1's own arithmetic, because a GNN fit
   is minutes where a tree is seconds. Widen `DATASETS` in the script if the budget
   allows; do not quietly assume the tree models' 5 x 9 grid was run here.

Remaining tasks (4–13) are largely independent loops over the existing pipeline. Three of
them already have work on disk:

- **Task 4** — `notebooks/LouvainEdgeSeverance.ipynb` quantifies the % of edges severed
  on every dataset, but only at `resolution = 10`. The resolution sweep, the
  split-pattern diagnostic and the no-Louvain run are still open.
- **Task 5** — code DONE, appendix runs in progress; full design in
  `GARG-AML_code_changes.md` §5. Two separable pieces:

  *The view mechanism.* A view is filtered on the **bank fields**
  (`From Bank == b or To Bank == b`), which selects a transaction set identical to
  client-membership filtering (verified, 0 differing rows) at a fraction of the cost.
  The **evaluated population is the bank's own clients**, which makes their labels
  identical to the full-data labels — a bank sees every transaction of its own clients —
  so only the features degrade. A view is addressed as a **dataset string**
  `<dataset>_bank<b>`, already threaded into every output path, so `evaluation.py`,
  `features.py` and `naming.py` need no changes and the full-data filenames stay
  byte-identical. `bank_views.resolve_banks` also expands a `"top<k>"` group spec into
  the k largest banks pooled as one institution.

  *The appendix experiment* (`scripts/partial_observability.py`) compares the **pure
  GARG-AML score** on the full graph vs the bank view, for the institution's clients
  only, **with Louvain off on both sides** — otherwise the two regimes are reduced by
  different partitions and the degradation cannot be attributed to missing edges. That
  makes its `full` column deliberately *not* the published Tables 10-11 number; say so
  wherever it is reported. Four measured facts to carry forward:
  - **N₁ survival is exactly 1.000 and N₂ survival is 0.069** (median 0.0099) on bank
    `012`; 53% of its clients lose their entire second-order neighbourhood. An earlier
    figure of ~21% was wrong — it counted everything within distance 2, so it included
    the fully-preserved N₁.
  - **The view biases the score upward**, it does not merely add noise: on `degree >= 3`
    clients the mean score rises +0.206 → +0.451, because losing N₂ empties the
    penalty blocks. More false positives, not just worse ranking.
  - **There is no "smallest bank"** worth running — 6,193 of 30,528 HI-Small banks have
    one client — and the *largest* holds only 0.512% of accounts, which is why the
    pooled `top50` (10.8%) carries the detection claim.
  - **The score is heavily tied** on one institution's clients (522 of `012`'s 2,639
    share exactly 1.0), so a raw top-K overlap between regimes is an artefact of sort
    order. Report `ties@K` beside every ranking metric; the module already computes it.
- **Task 12** — `src/utils/naming.py` is the canonical name map, used by the
  `DistributionScores`, `VisualisationResults` and `VisualisationRunTime` notebooks and
  now by `scripts/gargaml_tree.py` (via `gargaml_key()` / `pretty_config()`). Task 3 added
  `pretty_config()` for the ablations — note that the `topology` model is deliberately
  **not** labelled "GARG-AML" (it contains none). Task 1 added `graphsage_u` plus
  `MODEL_CONFIG_DISPLAY_NAMES`, which lets a model whose feature configs are its own
  thing (rather than a task-3 ablation) carry deliberate labels; it is checked before
  the `topology` rule, so "GraphSAGE (topology)" does not collide with the
  degree-only tree ablation's name.

**Notebook outputs are stripped automatically (task 13).** `.gitattributes` declares a
`nbstrip` clean filter for `*.ipynb`, implemented by `scripts/nbstrip.py` (standard
library only, so the filter has no install step). Git does not let a repository set the
filter *command* itself, so each clone registers it once:

```bash
git config filter.nbstrip.clean "python scripts/nbstrip.py --filter"
git config filter.nbstrip.smudge cat
```

This is already configured in the working clone, and the six tracked notebooks were
renormalised through it (2.7 MB → 126 KB). Working copies keep their rendered outputs;
only what git stores is stripped. `python scripts/nbstrip.py --check notebooks/` reports
what is not clean — note it also walks the gitignored exploratory notebooks, and the
in-place mode would strip those too, so prefer `--check` unless you mean it.

### Settings that are becoming parameters

Several values below are hard-coded today and are *being made configurable* by the
revision. Do not add new hard-coded uses of them:

| Value | Status |
|---|---|
| Louvain `resolution = 10` | Already a keyword argument (`graph_community(G, resolution=10)`); every call site takes the default and `LouvainEdgeSeverance.ipynb` hardcodes 10. Task 4 sweeps it — thread the value through the call sites, do not add new literals |
| 70/30 single split | Task 7 replaces it with **5-fold stratified CV on the IBM data only** — split logic must offer both `holdout_split` (synthetic, unchanged) and `cv_splits`; the fold partition is persisted to `results/<dataset>_folds.csv` and reused by GraphSAGE |
| Label cut-off list | Task 2 **done** — the threshold-free ranking metrics (P@K, R@K, lift@K, TP@K, AP) are reported alongside the swept cut-offs; report through `evaluation.py`, do not add a parallel metric path |
| Full-graph view (no bank filter) | Task 5 adds single-bank views — `construct_IBM_graph(..., banks=None)` and `define_ML_labels(..., banks=None)` keep today's behaviour by default; a view is named `<dataset>_bank<b>` (or `<dataset>_banktop<k>` for a pooled institution) and flows through as the `dataset` string. Do not special-case views downstream |
| Model names (`gargaml tree undirected` vs `GARG-AML Undir. Tree`) | Task 12 standardises — `src/utils/naming.py` is the one canonical scheme; route new labels through `pretty()`, or `pretty_config()` when a task-3 feature config is involved |

### Stale results on disk — every directed synthetic measure file

Commit `c5fba86` (2025-10-31, "bug fix") changed `measure_12_function`: an empty
block used to return density **1** unconditionally, and now returns 1 only when
`size_2 > 0`, else 0. `measure_12` is one of the two blocks Eq. 14 expects to be
*dense*, so the old behaviour handed full credit to nodes with no level-2
neighbours at all — inflating the directed score for exactly the nodes that are
least smurfing-like.

**Every directed synthetic measure file on disk still holds the pre-fix values**,
in all three results directories:

| Directory | Files | Date | Values |
|---|---|---|---|
| `results/` | 10 | Jan / Apr 2025 | pre-fix |
| `results-0/` | 66 | Apr 2025 | pre-fix |
| `results-3/` | 66 | Nov 2025 | pre-fix — **byte-identical to `results-0`**, so copied, not recomputed |

Do not trust the file dates: `results-3` postdates the fix and still contains the
old numbers. Verified by value, not by timestamp — the archives carry
`measure_12 = 1.0` where the current code gives `0.0`, on 67–84 % of nodes in the
datasets checked.

Effect on the directed score (`weighted_average`, three 100-node datasets):
scores shift **down** by 0.04–0.13 on average (up to 0.70 for one node), Spearman
correlation with the old ranking is 0.88–0.99, but **top-20 overlap is only
6/20 to 19/20** — the head of the ranking, which is what the tree models and the
alerting metrics use, moves materially on some datasets.

The undirected files are unaffected; the bug was directed-only.

**This bears on task 6.** The open question is why the directed score
underperforms the undirected one. A defect that inflated directed scores for
non-smurfing nodes, fixed in code in October 2025 but never reflected in any
stored result, is a plausible contributor — every directed synthetic number
reported so far was computed with it. Regenerating before instrumenting Eq. 11
is worthwhile.

**Decided (2026-09-21): keep the fix, regenerate the results.** `c5fba86` stays;
the directed synthetic measures are to be recomputed so the stored numbers match
the code. Do **not** revert `measure_12_function` to make the archives match.

Regeneration cost, from `results-0`'s own `time_results_dir*.txt`:

| Size | Datasets | Median | Subtotal |
|---|---|---|---|
| 100 | 22 | seconds | negligible |
| 10,000 | 22 | 4.1 min | ~1.6 h |
| 100,000 | 22 | 9.8 h | **~175–210 h** |

So 44 of the 66 datasets cost under two hours and the remaining 22 cost about
nine days of wall-clock — roughly 14 jobs against the 16 h VSC budget. Do the
100 and 10,000-node tiers first and see whether the directed average rank moves
at all before committing to the large tier. Note the Friedman/Nemenyi test is
specified over N=66, so a partial regeneration cannot be the final reported
number; it is a decision aid.

Then re-run downstream: `gargaml_tree_synthetic{,_3,_5}.py`, and rebuild the CD
diagrams and rank tables in `VisualisationResults.ipynb`. Back up `results*/`
first — the measure scripts overwrite in place.

**Unexplained, and worth checking before citing any runtime.**
`results-3/time_results_dir.txt` records 18 s – 10 min for the same
100,000-node datasets that `results-0` recorded at 8–11 h, while
`results-3`'s directed measure CSVs are byte-identical to `results-0`'s. Those
two facts cannot both describe one run: either `results-3`'s files were copied
in and its timings belong to something else, or its run did far less work. If
the runtime figure (Fig. 6) or any scalability claim draws on `results-3`,
confirm which run produced the numbers.

### Do not "fix" these

- The `try/except` around model training in `gargaml_tree.py` fills metric matrices
  with `NaN` when a (cutoff, pattern) cell has too few positives. That is **expected
  behaviour**, and the revision requires those gaps to be *reported*, not hidden.
- The GNN baseline must **never** receive GARG-AML scores, block densities, or block
  sizes as features. That would destroy the ablation's meaning.

### Fixed defect — do not reintroduce

`evaluate_model` used to compute AUC-ROC and AUC-PR from `clf.predict()` (hard 0/1
labels) rather than `predict_proba`, in `gargaml_tree.py`, `gargaml_tree_blocks.py` and
all three `gargaml_tree_synthetic*.py`. **Fixed in one place** by task 2:
`src/utils/evaluation.py::model_scores` (`predict_proba[:, 1]`, falling back to
`decision_function`, then `-score_samples` for the isolation forest — outlier detectors
are matched *before* `decision_function`, because `IsolationForest` exposes both). All
five call sites now go through it; any new evaluation path must take scores, not labels.

Two consequences to keep in mind. The fix *delivers* one of the metrics the reviewers
ask for rather than adding it: `average_precision_score` already **is** average
precision, so it is the existing `AUC_PR` column computed correctly, not a new column.
And AUC numbers produced before the fix are **not comparable** with numbers after it —
same column name, different quantity.

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
`random_state=1997` — including the `DecisionTreeClassifier`, which was missing it in
`gargaml_tree.py` until task 3 (sklearn breaks tied splits at random, so an unseeded tree
is not reproducible run to run; any tree result written before that fix is not comparable). Keep these fixed when comparing runs. Task 7's 5-fold CV keeps 1997
as the `StratifiedKFold` shuffle seed, so the partition is reproducible.

**Testing changes to the CV pipeline: use `N_FOLDS = 2`, not 5.** `gargaml_tree.py`'s
`N_FOLDS` module constant (0 = original single 70/30 split, ≥2 = that many stratified
folds) accepts any value ≥2 and produces a genuine, correctly-stratified partition either
way — it is not specific to 5. A real HI-Small run showed 5-fold CV can take tens of
minutes to hours depending on the model and sweep size, so when verifying a code change
(not producing paper numbers), set `N_FOLDS = 2` and keep `CUT_OFFS`/`TARGET_COLUMNS` to
a small subset first. `scripts/graphsage_baseline.py` reads whichever fold count is
actually present in `results/<dataset>_folds.csv` — it has no `N_FOLDS` of its own to
keep in sync, so regenerating that file with `gargaml_tree.py`'s `N_FOLDS = 2` is enough
to make a GraphSAGE test run fast too.

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

Range **[−1, 1]**. Eq. 14 is the whole score: there is **no max-with-transpose
step** in §3.3. Reverse flow is handled one level up, in the level assignment —
Eq. 11 plus "we repeat this selection on the reversed network", so a node at
distance two sits at level 0 when no directed path of length two reaches it in
*either* direction. That is what the `nodes_2_s` / `nodes_2_rs` set difference in
`GARG_AML_nodeselection_directed` implements.

`GARG_AML_node_directed` additionally takes `max(score, score_transposed)`. That
extra step has no counterpart in the paper, and it is **dead code**: its only
caller is the unused serial `GARG_AML(G)`. The pipeline scores through
`define_gargaml_scores` and uses the `GARGAML` column, which is Eq. 14 exactly.
Do not "restore" the max, and do not treat it as the paper's reverse-flow
handling — verified against §3.3 on 2026-09-18, and the two quantities differ
for 205 of 245 nodes on a 245-node synthetic graph, so the distinction is not
cosmetic.

**`GARGAML_max` is two different columns.** `define_gargaml_scores_directed`
emits it as the transpose-max; `combine_GARG_AML` emits it as the *neighbour*
max. In `gargaml_tree.py` the second overwrites the first, which is harmless
because only the neighbour max is ever a feature — but never assume which one a
frame holds.

> Reviewer note (see `GARG-AML_review_feedback.md`, R2 major 2): the undirected score
> outperforms the directed one, and task 6 investigates why — either benign
> bidirectional edges are over-penalised, or the level-assignment rule (Eq. 11) is too
> strict. Expect to instrument this code. The missing transpose-max is **not** a
> candidate explanation: it is absent from the paper and was never in the
> pipeline. Eq. 11 is where all the directional logic sits, so that is the more
> promising of the two hypotheses to instrument.

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
    graph_construction.py     # CSV -> NetworkX graph (bank= selects a single-bank view)
    pattern_construction.py   # parse *_Patterns.txt -> per-node AML labels
    synthetic_smurfing.py     # generate synthetic graphs (igraph) + injected patterns
    bank_views.py             # single-bank views (task 5): view_name(), bank_mask(),
                              #   bank_clients(), chunked client_counts() for LI-Large
    dataprep_vsc.py           # split/recombine the huge LI-Large CSV
  methods/
    GARGAML.py                # core: per-node block measures + score (entry: GARG_AML)
    gargaml_scores.py         # block measures -> "basic" / "weighted_average" scores
    graphsage.py              # GraphSAGE baseline (task 1): structure + feature build,
                              #   model, early-stopped training, timed inference
    utils/
      measure_functions_undirected.py   # the 3 undirected block-density measures
      measure_functions_directed.py     # the 9 directed block-density measures
      neighbourhood_functions.py        # node ordering, neighbour stats, final DataFrame
  utils/
    graph_processing.py       # graph_community() (Louvain filter), graph_degree() (hub removal)
    naming.py                 # canonical model names (task 12): MODEL_DISPLAY_NAMES, pretty(),
                              #   pretty_config() for the task-3 ablation labels
    hyperparameters.py        # hyperparameter provenance (task 9): every estimator's
                              #   configuration, sklearn defaults read at runtime
    evaluation.py             # shared metrics/splits/result writing (task 2)
    features.py               # the four feature groups + configs (task 3): feature_columns(),
                              #   is_direction_free(), feature_schema()

scripts/                      # runnable entry points — run from repo ROOT
  gargaml_directed.py         # directed measures on IBM data
  gargaml_undirected.py       # undirected variant
  gargaml_directed_synth.py   # directed measures on the synthetic grid
  gargaml_undirected_synth.py # undirected synthetic variant
  gargaml_tree.py             # decision tree + boosting on IBM scores
  gargaml_tree_blocks.py      # block-only ablation (task 3) — thin entry point on the
                              #   shared path in gargaml_tree.py
  gargaml_tree_synthetic.py   # tree models on synthetic data
  gargaml_tree_synthetic_3.py # …with 3 injected patterns
  gargaml_tree_synthetic_5.py # …with 5 injected patterns
  gargaml_IF.py               # isolation forest (unsupervised)
  gargaml_link_label.py       # edge-/link-level labelling
  partial_observability.py    # task 5 appendix: pure GARG-AML score, full graph vs
                              #   single-bank view, no Louvain, clients only
  distribution_scores.py      # score-distribution analysis/plots
  test_parallel.py            # multiprocessing sanity check, not part of the pipeline
  nbstrip.py                  # notebook-output clean filter (task 13), not part of the
                              #   pipeline; stdlib only, see §8

notebooks/                    # exploratory analysis & paper figures (run from repo root)
                              # only 7 are tracked; the exploratory ones are gitignored
                              # by name — see .gitignore
                              # BankObservability.ipynb is task 5's bank-size analysis
requirements.txt              # networkx>=3.0, pandas, numpy, scikit-learn, matplotlib,
                              # tqdm, igraph, torch, torch-geometric (task 1)
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
  critical-difference diagrams. Task 2 has added the ranking metrics (`src/utils/evaluation.py`); task 8 adds the χ²/p-value
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
`FigureA1`–`FigureA6`), the smurfing diagram, and the seven notebooks that produce paper
figures/analysis (`toyexample`, `VisualisationResults`, `VisualisationRunTime`,
`VisualisationNetwork`, `DistributionScores`, `LouvainEdgeSeverance`,
`BankObservability`). Ignored: all of
`data/`, `results*/` and `res/`, trained models (`*.pkl`), GraphViz exports (`*.dot`),
scratch plots, caches, `.DS_Store`, `initial_code.py`, the exploratory notebooks
(`AnalysisData`, `AnalysisParallel`, `AnalysisSyntheticData`, `AnomalousPatterns`,
`TestCode`, `Transfer`), and the reviewer/revision notes
(`GARG-AML_review_feedback.md`, `GARG-AML_code_changes.md`) — these are internal and
must not be pushed. Rationale is inline in `.gitignore`.

The README now carries a reproduction map (task 13): a "Reproducing the results" section
mapping each script/notebook to the files it writes, plus the experimental settings
(seeds, the `N_FOLDS` switch, feature configs, hyperparameter provenance). Keep it
current when adding experiments. Two things there are inferred rather than verified —
the figure numbers for the CD diagrams (Figs. 10–11) and the runtime boxplot (Fig. 6)
were matched by file size and timestamp, and the review notes refer to the CD diagrams
as Figs. 8–9 from the earlier round; confirm against the submitted manuscript. Still
open: the manuscript itself claims "Code ✓" without a repo link (R2 minor 4).
