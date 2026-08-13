# GARG-AML — Rejection & Reviewer Feedback (JMLC)

## Decision
- **Manuscript:** JMLC-11-2025-0207.R1 — *"GARG-AML against Smurfing: A Scalable and Interpretable Graph-Based Framework for Anti-Money Laundering"*
- **Journal:** Journal of Money Laundering Control
- **Outcome:** Rejected after R1 (no further review)
- **Editor:** Dr. Paul Gilmour (Editor-in-Chief)

### Editor's rationale
- Reviewers judged the manuscript below the publication bar and/or that revisions did not adequately address earlier concerns.
- **Scope mismatch:** too much weight on technical/data-science methods; insufficiently novel contribution to *criminological, regulatory, legal or policy* scholarship.
- Suggested fit: a specialist journal in **data science, computing, or finance**.

> Note: this is primarily a *scope* rejection, not a fundamental soundness rejection. The reviewer rated the core research as "competent" and the topic as valuable to practitioners.

---

## Actionable reviewer feedback

### 1. Confounded experimental design / missing ablation baseline
- Tree extensions (`gargaml tree`, `gargaml boost`) received neighborhood summary stats (min/mean/max/std of neighbor degrees and scores) that baselines did **not** get → cannot isolate the contribution of the block-density layout vs. standard local degree stats.
- **Action:** add a *topology-only* tree/XGBoost baseline trained on the same neighborhood summary stats but with all GARG-AML scores, block densities, and block sizes excluded.
- **Action:** explicitly document the exact structure and dimensions of the final feature matrix fed to the tree models (appendix acceptable).

### 2. Louvain edge-discarding blind spot
- Pipeline drops all inter-community edges before computing ego densities → may obscure multi-mule rings that deliberately cross communities/banks/rails.
- **Action:** quantify the **% of total edges severed** by the Louvain step and discuss the impact on detection coverage (appendix acceptable).

### 3. Unweighted binary adjacency matrix
- Treating a \$10 transfer the same as a \$50,000 wire strips signals central to structuring (amounts just under reporting thresholds).
- **Action:** expand discussion/limitations on extending block-density formulas to node/edge features (transaction volume) and **time-windowing**.

### 4. Statistical reporting & clarity
- Section 4.4 describes Friedman + post-hoc Nemenyi, but results omit the computed values.
- **Action:** report Friedman **χ²** and **p-values** per pattern category (appendix acceptable).
- **Action:** clarify the exact test implementation, and how **multiple-testing / Type I error** is controlled across the **N=66** synthetic datasets.
- **Action:** clarify how the Critical Difference cliques in **Figures 8 & 9** are derived.

### 5. Evaluation alignment with method intent
- GARG-AML targets scatter-gather / gather-scatter smurfing geometry, but is evaluated on IBM's pooled **"Is Laundering"** label (bundles 8 distinct patterns: fan-out, cycles, random, bipartite, etc.), which may dampen AUC-PR/AUC-ROC.
- **Action:** justify evaluating a pattern-specific detector against a catch-all label; ideally **isolate and report metrics against the simulated smurfing/structuring patterns**.

### 6. Naming inconsistencies
- e.g. `gargaml tree undirected` (Table 9) vs. `GARG-AML Undir. Tree` (Tables 10–11).
- **Action:** standardize model names across all text, tables, and figures.

### 7. Literature review gaps (Section 2)
Add recent graph-based/GNN AML work, including:
- Effendi & Chattopadhyay (2024), *Privacy-preserving graph-based ML with FHE for collaborative AML*, SPACE.
- Johannessen & Jullum (2025), *Finding money launderers using heterogeneous GNNs*, J. Finance and Data Science.
- Khanvilkar & Kommuru (2025), *Regulatory graphs and GenAI for real-time transaction monitoring*, arXiv:2506.01093.
- Rashid & Hayat (2025), *AMLGaurd: Graph-Based Money Laundering Detection*, ICECCE (IEEE).

### 8. Typos flagged (likely more)
- P2L-1: "Additionally, The volume…"
- P3L-8: "…identifies anomalous sub-graphs…"
- P4L-1: "This elements of A…"
- P13L11: "…four additional model."

---

## Reviewer's overall verdict
- **Strengths:** white-box egocentric design; O(|V|) on sparse graphs; deployable in SAR workflows without black-box GNN risk; large-scale (multi-million-edge) testing.
- **Blockers:** experimental confounding (feature imbalance), unreported statistical test values + ambiguous setup, unquantified Louvain edge-loss, literature gaps.
- Recommended "accept only after revision" addressing the above — overruled by the editor's scope decision.
