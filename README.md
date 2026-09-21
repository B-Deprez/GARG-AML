# **GARG-AML:** *finding smurfing using Graph-Aided Risk Guarding for Anti-Money Laundering* </br><sub><sub>*Bruno Deprez, Bart Baesens, Tim Verdonck, Wouter Verbeke* </sub></sub>

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

This is the source code for an experiment to detect smurfing patterns in transaction networks. It provides an implementation of GARG-AML, which constructs a score based on the adjancy matrix of the second-order neighbourhood. 

## Methodology
GARG-AML is based on insights derived from the definition of a pure smurfing pattern. With smurfing, multiple intermediate money mules (or smurfs) are used to get a large amount of money from one account to another, often using many small transactions. A representation of this is given in the figure below. 

![Smurfing network](./assets/img/SmurfingNetwork.png)

Translating this figure into a adjacency matrix for the second order neighbourhood, gives us the following:
$$\begin{array}{r}
        A \\ E \\ B \\ C \\ D
    \end{array}
    \begin{pmatrix}
         0 & 0 & 1 &1 &1\\
         0 & 0 & 1 &1 &1\\
         1 & 1 & 0 &0 &0 \\
         1 & 1 & 0 &0 &0 \\
         1 & 1 & 0 &0 &0 \\
    \end{pmatrix}$$

We can clearly distinguish four blocks in the adjacency matrix. For a typical smurfing pattern, the on-diagonal blocks only contain $0$, while the off-diagonal blocks are fully populated with $1$'s. The GARG-AML scores are calculated based on the density of these blocks. 

## Data 
The experiments are evaluated on synthetic data which is made publically available. 

The repository does not provide any data, due to size constraints. The data can be found online using the following link:
- [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

## Experimental Evaluation
GARG-AML is tested against the current state-of-the-art, namely Flowscope [1] and AutoAudit [2]. The code of these two models is taken from the respective repositories and not included in this one. We refer the interested coder to the corresponding forked repositories for [Flowscope](https://github.com/B-Deprez/flowscope) and [AutoAudit](https://github.com/B-Deprez/AutoAudit), which include changes made to analyse the data sets included in this study. The code for analysing the output of the SOTA on the other hand is provided. 

## Reproducing the results

Every script is run **from the repository root** (each one does `os.chdir("./")`
and uses root-relative paths, so running from inside `scripts/` breaks them).
The pipeline is two-staged: a measure script writes per-node block measures to
`results/`, and a model script reads them back to train and evaluate.

```bash
python scripts/gargaml_undirected.py   # stage 1: block measures -> results/
python scripts/gargaml_tree.py         # stage 2: train + evaluate -> results/
```

### Which script produces which result

| Result | Produced by | Output |
| --- | --- | --- |
| GARG-AML block measures, IBM data | `scripts/gargaml_undirected.py`, `scripts/gargaml_directed.py` | `results/<dataset>_GARGAML_<direction>.csv` |
| GARG-AML block measures, synthetic grid | `scripts/gargaml_undirected_synth.py`, `scripts/gargaml_directed_synth.py` | `results/<dataset>_GARGAML_<direction>.csv` |
| Tree / boosting results on the IBM data | `scripts/gargaml_tree.py` | `results/<dataset>_<direction>_metrics.csv` (tidy) and `results/<dataset>_<metric>_<model>_<direction>_combined.csv` |
| Feature-group ablations (blocks, topology, all) | `scripts/gargaml_tree.py`, `scripts/gargaml_tree_blocks.py` | the same files with a `_blocks` / `_topology` / `_all` suffix; the published model is the suffix-less `full` config |
| Tree / boosting results on the synthetic grid | `scripts/gargaml_tree_synthetic{,_3,_5}.py` | `synthetic_tree_<directed>_{full,3,5}.csv` (repository root) |
| Isolation-forest baseline | `scripts/gargaml_IF.py` | `results/<dataset>_<direction>_if_metrics.csv` and `results/<dataset>_<metric>_isolationforest_<direction>_if_combined.csv` |
| GraphSAGE baseline, both feature configs | `scripts/graphsage_baseline.py` | `results/<dataset>_undirected_graphsage[_attr]_metrics.csv`, the matching `_combined.csv` matrices, plus `results/<dataset>_graphsage_runs.csv` (one row per run, with timings) and `_graphsage_summary.csv` (mean/std over folds) |
| Score distributions, histograms and lift curves | `scripts/distribution_scores.py`, `notebooks/DistributionScores.ipynb` | `results/<dataset>_GARGAML_<direction>_*histogram.pdf`, `*_lift.pdf` |
| Base GARG-AML score metrics, per fold and pooled | `scripts/distribution_scores.py` | `results/<dataset>_<direction>_base_metrics.csv` (tidy; the full-population row keeps `fold = NaN` and is the number in `results/results_performance_IBM_<direction>.txt`, unchanged) |
| Performance tables and figures | `notebooks/VisualisationResults.ipynb` | `results/<dataset>_AUC-ROC_AUC-PR.pdf`, LaTeX tables |
| Critical-difference diagrams (Figs. 10-11) | `notebooks/VisualisationResults.ipynb` | `results/CD_ROC_full.pdf`, `results/CD_PR_full.pdf` |
| Friedman &chi;&sup2; / p-values, multiple-testing control, Nemenyi matrices | `notebooks/VisualisationResults.ipynb` | `results/friedman_results.csv` (per metric and pattern: &chi;&sup2;, df, raw *p*, Bonferroni- and Holm-adjusted *p*, the Nemenyi critical difference) and `results/nemenyi_pvalues_<metric>_<pattern>.csv` |
| Runtime / scalability comparison (Fig. 6) | `notebooks/VisualisationRunTime.ipynb` | `results/time_boxplot_norm.pdf` |
| Synthetic network illustrations | `notebooks/VisualisationNetwork.ipynb` | `data/combined_synthetic_networks.pdf` |
| Worked toy example (Appendix A) | `notebooks/toyexample.ipynb` | inline figures |
| Edges severed by the Louvain filter | `notebooks/LouvainEdgeSeverance.ipynb` | inline table |
| Louvain sensitivity sweep (resolution, and no Louvain at all) | `scripts/gargaml_undirected.py`, `scripts/gargaml_directed.py`, `scripts/gargaml_tree.py`, run on the `_res<r>` / `_nolouvain` dataset names | `results/louvain_severance.csv` (edges severed per dataset and setting) plus the usual per-dataset measure and metric files under those names |
| Partial-observability appendix: score on the full graph vs a bank's view | `scripts/partial_observability.py` | `results/<view>_partial_observability_accounts.csv`, `..._metrics.csv` |
| Appendix tables and figures | `notebooks/BankObservability.ipynb` | `results/appendix_*.csv`, `results/appendix_*.pdf` |
| Directed-vs-undirected diagnosis | `scripts/directed_diagnosis.py` | `results/<dataset>_directed_diagnosis.csv` (per node: level census, reciprocal census, five score variants), `..._summary.csv` (means by ground-truth class and structural role), `..._directed_diagnosis_metrics.csv` (each variant through the shared metrics), and the pooled `results/directed_diagnosis_{summary,metrics}.csv` |
| Revision tables: results, ablation, alert queue, fold variance, cost, Louvain sweep and edge severance | `scripts/build_tables.py` | `results/table_*.tex` and a `.csv` twin of each, plus `results/table_coverage.csv` saying which model/config combinations are on disk and whether they predate the cross-validation |
| Tree / boosting / GraphSAGE under a bank view | the model scripts above, run on a view name | the same files, under `results/<dataset>_bank<b>_*` |

FlowScope and AutoAudit are not run from this repository (see *Experimental
Evaluation* above); `VisualisationResults.ipynb` reads their exported results
from `results-0/` and `results-aa/`.

### Experimental settings

- **Reproducibility.** Louvain uses `seed=1997` and every scikit-learn split and
  estimator uses `random_state=1997`. Keep these fixed when comparing runs.
- **Cross-validation.** `scripts/gargaml_tree.py` carries an `N_FOLDS` switch:
  `0` reproduces the original single stratified 70/30 split, `>= 2` runs that
  many stratified folds plus a pooled out-of-fold pass. The fold partition is
  written to `results/<dataset>_folds.csv` so other models (GraphSAGE, and the
  base scores in `scripts/distribution_scores.py`) evaluate on exactly the same
  folds. Note that the evaluation is **transductive**: the neighbourhood
  summary features are computed on the full graph before splitting.
  Cross-validation is scoped to the IBM datasets; the 66 synthetic datasets
  keep the single 70/30 split, and every consumer of the partition falls back
  to full-population metrics when no `_folds.csv` exists.
  Under `N_FOLDS >= 2` the historical `_combined.csv` matrices hold the **mean
  over folds** rather than a single split's value — same filenames and shape,
  a different quantity — with `_std_combined.csv` companions beside them.
- **Reading the tables.** `scripts/build_tables.py` assembles every table from
  the tidy metrics and never mixes the three kinds of row the `fold` column
  distinguishes: per-fold (`>= 0`) gives the mean and spread, pooled
  out-of-fold (`-1`) is the only correct population for the alert-queue
  metrics, and `NaN` is a single-split or full-population run. Alert tables
  print the population each row was ranked over, because a model still on the
  single split is ranked over its test slice while a cross-validated one is
  ranked over every account. A `--` is a cell that could not be evaluated and
  a starred cell is a mean over fewer folds than the rest; both are reported
  rather than dropped.
- **Louvain setting.** The pre-processing resolution rides in the dataset
  name, the same way a bank view does: `HI-Small_res20` is HI-Small reduced at
  resolution 20, `HI-Small_nolouvain` skips the reduction entirely, and a bare
  `HI-Small` keeps the published value of 10, so existing result files are
  untouched. Every run prints the percentage of edges severed and stage 1
  appends it to `results/louvain_severance.csv`. The no-Louvain arm is far more
  than a slower run: the reduction is what bounds the second-order ego graphs,
  and without it a single HI-Small node can densify to roughly 1.8 GB.
- **Per-dataset sweep reductions.** `DATASET_SETTINGS` in
  `scripts/gargaml_tree.py` narrows the cut-off/pattern grid for one dataset
  without touching the defaults: LI-Large runs the paper's headline cut-offs
  (0.1 / 0.5 / 0.9) only, matching `graphsage_baseline.py`, because the full
  grid is 360 fits and 1800 under 5-fold CV. Omitted cells are still written,
  as `NaN` with a `"not in this dataset's sweep"` status, so the result
  matrices keep their published shape and the reduction is visible in the
  output rather than inferred from a missing row.
- **Feature configurations.** `src/utils/features.py` defines four column
  groups and the named configurations that select them; `run_config` writes the
  exact matrix it used to
  `results/<dataset>_<direction><suffix>_feature_schema.csv`.
- **GraphSAGE baseline.** Two feature configurations are reported side by
  side: `topology` (degree and log-degree, strict parity with GARG-AML's
  inputs) and `attributes` (adds per-account amount, count, currency, bank and
  timing aggregates, deliberately generous). Neither ever receives a GARG-AML
  score, block density or block size. It reads the same
  `results/<dataset>_folds.csv` partition as the tree models, so the two are
  paired fold by fold, and early stopping uses a stratified 10% slice carved
  out of the *training* fold only. Preprocessing, fit and inference times are
  recorded separately, since GARG-AML's fit cost is zero and a single
  wall-clock number would hide that.
- **Partial observability.** Any script that takes a dataset name also accepts a
  **single-bank view** `<dataset>_bank<b>` (e.g. `HI-Small_bank012`): it reads the
  same `data/<dataset>_Trans.csv` but keeps only the transactions booked at bank
  *b*, i.e. those with at least one of its clients on them. The view is scored
  and evaluated on **the bank's own clients**; external counterparties stay in
  the graph as neighbours but are never scored. Because a bank sees every
  transaction of its own clients, their labels are identical to the full-data
  labels, so a view-vs-full comparison varies only the features. Views write to
  `results/<dataset>_bank<b>_*`, leaving the full-data files untouched. Run the
  measure script on the view before the model script, and choose the banks with
  `notebooks/BankObservability.ipynb`. Expect many `(cut-off, target)` cells to
  be reported as NaN: a single bank's view holds few positives.
- **Partial-observability appendix.** `scripts/partial_observability.py` compares the
  **pure GARG-AML score** computed on the full transaction graph against the same score
  computed on one institution's view, for that institution's clients. Louvain is applied
  to **neither** side, so the difference is attributable to the missing edges rather than
  to two different community partitions — which also means its `full` column is *not* the
  Louvain-reduced number reported in the main results. Only the institution's clients are
  scored, never the whole graph. An institution is one bank (`"012"`) or a `"top<k>"`
  group standing for the k largest pooled into one.
- **Hyperparameters.** No hyperparameter search was performed: every value is
  either a scikit-learn default or a single value fixed a priori, and none of
  them were selected using the test split. `src/utils/hyperparameters.py`
  records the full configuration, with the provenance of each value, to
  `results/<dataset>_hyperparameters.csv`. Regenerate it on its own with:

```bash
python -m src.utils.hyperparameters
```

## Repository structure
```
src/
  data/                       # data loading & generation
    graph_construction.py     #   transaction CSV -> NetworkX graph
    pattern_construction.py   #   parse *_Patterns.txt into per-node AML labels
    synthetic_smurfing.py     #   generate synthetic graphs with injected smurfing
    dataprep_vsc.py           #   split/recombine the large LI-Large CSV
  methods/
    GARGAML.py                # core: per-node block measures + GARG-AML score
    gargaml_scores.py         # turn block measures into summary scores
    utils/                    #   block-density measures (directed & undirected),
                              #   node ordering and neighbourhood statistics
    graphsage.py              # GraphSAGE baseline: graph build, model, training
    directed_diagnosis.py     # why undirected beats directed: level census,
                              #   reciprocal-edge census, five score variants
  utils/
    graph_processing.py       # Louvain community filtering & hub removal
    evaluation.py             # shared metrics, splits and result writing
    features.py               # feature column groups and named configurations
    naming.py                 # canonical model names for tables and figures
    hyperparameters.py        # hyperparameter provenance for every fitted model
    reporting.py              # tidy metrics -> paper tables (fold-aware)

scripts/                      # runnable entry points (run from the repo root)
  gargaml_directed.py         #   compute directed measures on IBM data
  gargaml_undirected.py       #   undirected variant
  gargaml_*_synth.py          #   same, on the synthetic dataset grid
  gargaml_tree*.py            #   train/evaluate decision-tree & boosting models
  gargaml_IF.py               #   isolation-forest (unsupervised) variant
  graphsage_baseline.py       #   GraphSAGE baseline on the same folds
  gargaml_link_label.py       #   edge-/link-level labelling
  distribution_scores.py      #   score-distribution analysis
  build_tables.py             #   tidy metrics -> LaTeX/CSV tables for the paper
  directed_diagnosis.py       #   directed-vs-undirected diagnosis on the
                              #   synthetic grid (and sampled HI-Small)
  nbstrip.py                  #   repository hygiene, not part of the pipeline

notebooks/                    # exploratory analysis and paper figures
assets/                       # README images
data/                         # datasets (not tracked — see "Data" above)
results/, res/                # generated outputs (not tracked)
```

The typical workflow is two-staged: a `gargaml_*` script computes the GARG-AML
block measures and writes them to `results/<dataset>_GARGAML_<dir>.csv`, then a
`gargaml_tree*` / `gargaml_IF` script reads those scores back to train and
evaluate a classifier. Run every script from the repository root. See
[`CLAUDE.md`](./CLAUDE.md) for a fuller description of the method, data flow,
and conventions.

## Installing 
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

### Notebook outputs

Notebook outputs are kept out of git: they are large (figure data in one
notebook alone accounted for 2.1 MB) and they have previously caused pushes to
fail. `.gitattributes` declares a `nbstrip` clean filter for `*.ipynb`, but git
deliberately does not let a repository configure the command itself, so each
clone registers it once:

```bash
git config filter.nbstrip.clean "python scripts/nbstrip.py --filter"
git config filter.nbstrip.smudge cat
```

Your working copies keep their rendered outputs -- only what git stores is
stripped. The filter uses the standard library only, so there is nothing extra
to install. To check or strip files directly:

```bash
python scripts/nbstrip.py --check notebooks/
```

## Citing
Please cite our paper and/or code as follows:
*Use the BibTeX citation*

```tex

@article{deprez2025gargamlsmurfingscalableinterpretable,
      title={GARG-AML against Smurfing: A Scalable and Interpretable Graph-Based Framework for Anti-Money Laundering}, 
      author={Bruno Deprez and Bart Baesens and Tim Verdonck and Wouter Verbeke},
      year={2025},
      journal={arXiv preprint arXiv:2506.04292},
      eprint={2506.04292},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2506.04292}, 
}

```

## References
[1] Li, X., Liu, S., Li, Z., Han, X., Shi, C., Hooi, B., ... & Cheng, X. (2020). Flowscope: Spotting money laundering based on graphs. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 04, pp. 4731-4738).

[2] Lee, M. C., Zhao, Y., Wang, A., Liang, P. J., Akoglu, L., Tseng, V. S., & Faloutsos, C. (2020). Autoaudit: Mining accounting and time-evolving graphs. In 2020 IEEE International Conference on Big Data (Big Data) (pp. 950-956). IEEE.
