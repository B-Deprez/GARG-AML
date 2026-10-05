# Running GARG-AML on the VSC (Slurm)

One `.slurm` file per experiment. Every job selects its work through the
environment rather than through an edit to the source, so the `.out` file
reconstructs the run and 66 array tasks can share one checkout.

Each `.slurm` file opens with a header block giving what it runs, the exact
`sbatch` line, its prerequisite job, its `--array` mapping, whether it resumes,
and the arithmetic behind `--mem` / `--cpus-per-task`. **Read the header before
submitting** — several carry warnings that matter more than the resources.

```bash
bash slurm/common.sh            # print the dataset names and array ranges
bash slurm/common.sh --verify   # check that list against the Python source of truth
```

---

## Before the first submission

**Back up `results/` first.** The measure scripts overwrite in place, and
`results-0/` and `results-3/` are pre-fix archives whose provenance is already
uncertain — `results-3`'s directed synthetic CSVs are byte-identical to
`results-0`'s while its timing log records runtimes 30× shorter, so those two
files cannot both describe one run. Do not add a third ambiguous archive.

```bash
cp -a results results.bak.$(date +%F)
```

---

## How a job picks its work

Every run-scope constant stays in the script as the documented default and is
overridden by `GARGAML_<NAME>` for the duration of a job (`src/utils/runtime.py`).
Running a script bare behaves exactly as it always has.

| Variable | Effect |
|---|---|
| `GARGAML_DATASET` | one dataset name, used verbatim |
| `GARGAML_DATASETS` | comma-separated work list |
| `GARGAML_DATASET_INDEX` / `SLURM_ARRAY_TASK_ID` | index into the script's own list |
| `GARGAML_N_CPU` | worker-pool width: a fixed `min(4, cpu_count() // 2)` for the measure scripts, `SLURM_CPUS_PER_TASK` for `gargaml_tree.py`, where `1` means serial (see the caveat below) |
| `GARGAML_N_FOLDS` | `gargaml_tree.py` split protocol; `2` is the cheap test setting |
| `GARGAML_CONFIGS` | GraphSAGE feature configs (`topology`, `attributes`) |
| `GARGAML_INSTITUTIONS` | `partial_observability.py` bank view |
| `GARGAML_FORCE=1` | recompute even when the output CSV already exists |
| `GARGAML_REQUIRE_GPU=1` | turn the silent CPU fallback into an immediate failure |

A dataset string is already this repo's run tag — `_res<r>`, `_nolouvain`,
`_bank<b>`, `_banktop<k>` ride in the name and namespace every output file.
There is no second naming scheme.

---

## Submission order

Copy-pasteable. Stage 1 must precede stage 2 for the **same dataset name**, and
the tree job must precede GraphSAGE for that name because it writes the fold
partition GraphSAGE reads.

Two things bite when chaining these by hand. `sbatch --parsable` returns
`<jobid>;<cluster>` here, and `;` is a shell command separator, so an unquoted
id splits the sbatch line in two -- `slurm/submit.sh` strips the suffix, a bare
`sbatch` needs `| cut -d";" -f1`. And a dependency only holds while the job it
names is still known to the scheduler: once stage 1 has completed and left the
queue, submitting stage 2 against its id fails with `Job dependency problem`.
Drop the `--dependency` and gate on the files instead -- the LI-Large waves
below do exactly that, and see Recovery.

```bash
# --- Stage 1: IBM measures (2 jobs x 1 task each, CPU) ----------------------
# Index 4 = HI-Small, the published setting. See `bash slurm/common.sh`.
dir=$(slurm/submit.sh slurm/measures_ibm_dir.slurm   HI-Small --array=4)
und=$(slurm/submit.sh slurm/measures_ibm_undir.slurm HI-Small --array=4)

# --- Stage 1: synthetic measures (2 jobs x 22 tasks per tier, CPU) ----------
# Submit the tiers separately -- their costs differ by four orders of magnitude.
sd1=$(sbatch --parsable --array=0-21  --time=00:30:00 --mem=8g  slurm/measures_synth_dir.slurm | cut -d";" -f1)
sd2=$(sbatch --parsable --array=22-43 --time=02:00:00 --mem=16g slurm/measures_synth_dir.slurm | cut -d";" -f1)
su1=$(sbatch --parsable --array=0-21  --time=00:30:00 --mem=8g  slurm/measures_synth_undir.slurm | cut -d";" -f1)
su2=$(sbatch --parsable --array=22-43 --time=02:00:00 --mem=16g slurm/measures_synth_undir.slurm | cut -d";" -f1)
# The 100,000-node tier only once the cheap ones look right (22 tasks, ~10 h each):
# sd3=$(sbatch --parsable --array=44-65 --time=24:00:00 --mem=64g slurm/measures_synth_dir.slurm | cut -d";" -f1)

# --- Stage 2: models (CPU) --------------------------------------------------
tree=$(slurm/submit.sh slurm/tree.slurm HI-Small --array=4 --dependency=afterok:$dir:$und)
# tree_blocks.slurm and if.slurm hardcode their dataset in the Python script
# (main()), so submit them directly with sbatch, not through submit.sh.
blk=$(sbatch --parsable --dependency=afterok:$dir:$und slurm/tree_blocks.slurm | cut -d";" -f1)
ifj=$(sbatch --parsable --dependency=afterok:$dir:$und slurm/if.slurm | cut -d";" -f1)
# One job per (variant, direction) -- the argument has no default direction.
# VisualisationResults.ipynb reads the 3 and 5 variants; $ts joins the four ids.
ts3u=$(slurm/submit.sh slurm/tree_synth.slurm 3_undirected --dependency=afterok:$sd1:$su1)
ts3d=$(slurm/submit.sh slurm/tree_synth.slurm 3_directed   --dependency=afterok:$sd1:$su1)
ts5u=$(slurm/submit.sh slurm/tree_synth.slurm 5_undirected --dependency=afterok:$sd1:$su1)
ts5d=$(slurm/submit.sh slurm/tree_synth.slurm 5_directed   --dependency=afterok:$sd1:$su1)
ts=$ts3u:$ts3d:$ts5u:$ts5d
# The base score on the synthetic grid (both directions, one job):
sds=$(slurm/submit.sh slurm/distribution_scores.slurm synthetic --dependency=afterok:$sd1:$su1)

# --- Stage 2: GraphSAGE (1 job, GPU) ---------------------------------------
# afterok on the TREE job: it consumes results/<dataset>_folds.csv.
gs=$(slurm/submit.sh slurm/graphsage.slurm HI-Small --dependency=afterok:$tree)

# --- Appendices / diagnostics (CPU, independent of stage 1) -----------------
ps=$(slurm/submit.sh  slurm/pattern_splitting.slurm HI-Small)
po1=$(slurm/submit.sh slurm/partial_obs.slurm 012)
po2=$(slurm/submit.sh slurm/partial_obs.slurm top50)
dd=$(sbatch --parsable slurm/directed_diagnosis.slurm | cut -d";" -f1)
# Tables 7-8's label-percentage matrices -- reads only data/, not fed into
# collect.slurm (build_tables.py doesn't touch these; DistributionScores.ipynb
# reads the CSV directly).
ld=$(slurm/submit.sh  slurm/label_distribution.slurm HI-Small)

# --- Reporting (1 core, minutes) -------------------------------------------
# afterANY, so one failed arm does not block the tables: build_tables.py
# announces missing tables and renders unfittable cells as "--" by design.
slurm/submit.sh slurm/collect.slurm all \
  --dependency=afterany:$tree:$blk:$ifj:$ts:$gs:$ps:$po1:$po2:$dd
```

### LI-Large, submitted in waves

LI-Large is the documented 16 h / 200 GB job and needs explicit overrides (the
tree job's `--cpus-per-task=36` is already its default, repeated so the line
states the whole allocation). Submit it **in waves, without `--dependency`**:
each wave is gated on the files the previous one wrote, not on a job id. That
way every line stands alone and can be run in its own cell or shell, because
no job id has to carry over from one line to the next. A chained submission
fails with `Job dependency problem` as soon as one of those ids is empty or
has left the queue.

Run from `$VSC_DATA/GARGAML/11.2Code/GARG-AML` after a `git pull`. The pipeline
reads `data/LI-Large_Trans.csv` and `data/LI-Large_Patterns.txt` whole, under
exactly those names. The `_0` … `_29` pieces that `src/data/dataprep_vsc.py`
splits off are read by nothing, so skip that step.

```bash
# --- Before wave 1: is stage 1 already queued? ------------------------------
# If both measures_ibm_*_LI-Large jobs are listed, do not submit them again --
# two copies both do the full 16 h of work.
squeue -M wice -u $USER -o "%.14i %.32j %.9T %.11M"

# --- Wave 1: now. Stage 1, plus the jobs that read only data/ ---------------
slurm/submit.sh slurm/measures_ibm_dir.slurm   LI-Large --array=7 --time=16:00:00 --mem=200g
slurm/submit.sh slurm/measures_ibm_undir.slurm LI-Large --array=7 --time=16:00:00 --mem=200g
# label_distribution peaked at 166.6 GiB / 38 min (job 62206918); pattern
# splitting's LI-Large read is still unmeasured -- same headroom until timed.
slurm/submit.sh slurm/label_distribution.slurm LI-Large --time=16:00:00 --mem=200g
slurm/submit.sh slurm/pattern_splitting.slurm  LI-Large --time=16:00:00 --mem=200g
# GraphSAGE's labels, structure and features, built on a CPU node: the label
# build alone exceeds what a gpu_a100 job may request (126,000 MiB per GPU).
slurm/submit.sh slurm/graphsage_prep.slurm     LI-Large --time=06:00:00 --mem=200g

# --- Wave 2: once BOTH measure files exist ----------------------------------
ls -lh results-revision/LI-Large_GARGAML_*.csv
slurm/submit.sh slurm/tree.slurm LI-Large --array=7 --cpus-per-task=36 --time=16:00:00 --mem=200g

# --- Wave 3: once the fold partition exists ---------------------------------
# The tree job writes it after its first data preparation, long before it
# finishes, so these two can run alongside the tree fits. GraphSAGE also
# needs all four caches from graphsage_prep (prep.csv is written last):
# without them it rebuilds the labels on the GPU node and runs out of memory.
ls -lh results-revision/LI-Large_folds.csv results-revision/LI-Large_graphsage_*
slurm/submit.sh slurm/graphsage.slurm LI-Large --time=24:00:00 --mem=120g
sbatch --time=16:00:00 --mem=200g slurm/distribution_scores.slurm

# --- Wave 4: once squeue shows nothing left for LI-Large --------------------
# LI-Large_directed_all_metrics.csv is the tree job's last section; until it
# is listed, the tree job has not finished.
ls results-revision/LI-Large_*metrics.csv
slurm/submit.sh slurm/collect.slurm all
```

- **The wave-2 gate is the one that matters.** A tree job that starts before a
  measure file exists skips that direction and still exits 0.
- **GraphSAGE's preprocessing time is a cache load on LI-Large.** The GPU job
  reads what `graphsage_prep` built, so the `preprocess_seconds` it reports
  is flagged `preprocess_from_cache = 1`. The cold build times are in
  `LI-Large_graphsage_prep.csv`; use those for any runtime comparison.
- **Stage 1 skips a dataset whose measures CSV already exists** in
  `results-revision/`. Put `GARGAML_FORCE=1` in front of the two wave-1 lines
  to recompute it.
- **A 16 h kill loses a whole direction.** Stage 1 resumes per file, not per
  node. The 16 h is the budget the paper states, not a measured runtime, so if
  the limit is hit, resubmit with a longer `--time`.
- **`distribution_scores.slurm` with no argument runs HI-Small and LI-Large
  both** (an argument -- a dataset name, or `synthetic` -- narrows it to that
  one). Wave 3 submits it bare and therefore also rewrites
  HI-Small's base-score metrics, using whichever `HI-Small_folds.csv` is in
  `results-revision/` at that point.
- **Not in this run.** `tree_blocks.slurm` is redundant here, because
  `tree.slurm` already runs the `blocks` config. `if.slurm` would need an edit,
  since `gargaml_IF.py` hardcodes `dataset = "HI-Small"` in `main()`.
  `LI-Large_nolouvain` (index 9) is expected to be infeasible; see Known gaps.

---

## What depends on what

```
data/  ──┬─► measures_ibm_{dir,undir}   ──┬─► tree ──► graphsage   (folds.csv)
         │   (per dataset NAME)           ├─► tree_blocks
         │                                └─► if
         ├─► graphsage_prep ───────────────────► graphsage   (caches; LI-Large)
         ├─► measures_synth_{dir,undir} ────► tree_synth           (see gap below)
         ├─► pattern_splitting          ─┐
         ├─► partial_obs                 ├──► collect  (afterany)
         └─► directed_diagnosis         ─┘
```

Two dependencies are real and verified in the code:

- **Stage 1 → stage 2, same dataset name.** `gargaml_tree.py:183` keeps only the
  directions whose measures CSV exists and reports the gap, so a missing file is
  a *skip*, not a crash. `afterok` is the strict choice; `afterany` is defensible
  if you want the undirected arm to proceed when the directed one fails.
- **tree → GraphSAGE, same dataset name.** `gargaml_tree.py` writes
  `results/<dataset>_folds.csv` (only when `N_FOLDS >= 2`), and
  `graphsage_baseline.py:361` reads it for both the fold count and fold
  membership. GraphSAGE has no `N_FOLDS` of its own, so it inherits whatever the
  tree ran with — **a 2-fold test partition silently yields a 2-fold GraphSAGE
  run**. `results/HI-Small_folds.csv` currently holds folds `[0, 1]` from a
  verification run; regenerate it with `N_FOLDS=5` before producing paper numbers.

---

## Recovery

**Ground truth is what got written, not the exit code** — a task can exit 0
having written nothing, and `sacct` ages out.

The four measure scripts skip a dataset whose output CSV already exists, so
**resubmitting the same array is the recovery procedure**. Writes are atomic
(temp file in the destination directory, then `os.replace`), so a wall-time kill
cannot leave a truncated CSV that the skip would mistake for finished work.

```bash
# What is actually on disk, per direction:
ls results/*_GARGAML_directed.csv   | wc -l
ls results/*_GARGAML_undirected.csv | wc -l

# Resubmit a whole tier; finished datasets cost ~0.4 s each and say so.
sbatch --array=22-43 --time=02:00:00 --mem=16g slurm/measures_synth_dir.slurm

# Deliberately recompute (e.g. the pending post-c5fba86 regeneration):
GARGAML_FORCE=1 sbatch --array=44-65 --time=24:00:00 --mem=64g slurm/measures_synth_dir.slurm
```

Stage 2 and GraphSAGE do **not** resume at cell granularity. `gargaml_tree.py`
writes each direction/feature-config section's files when that section
finishes, so a wall-time kill loses only the section in progress — but nothing
skips finished sections, so a resubmission refits all of them. GraphSAGE's
metric writers run after the loops, not inside them: it checkpoints every epoch
and resumes mid-fold, but a wall-time kill still loses the metrics of every
completed fit while keeping their checkpoints. Size `--time` to finish.

Pending jobs, with the reason:

```bash
squeue -M wice -u $USER -o "%.14i %.26j %.9T %.11M %R"
```

The `REASON` column distinguishes genuine contention (`Priority`, `Resources`)
from a per-user GPU cap (`AssocGrpGpuLimit`), which no amount of waiting clears.

---

## Known gaps — read before trusting a rerun

**Synthetic stage 1 → stage 2 is now wired.** `gargaml_undirected_synth.py`
writes `<dir>/<ds>_GARGAML_undirected_parallel.csv` and
`gargaml_tree_synthetic{,_3,_5}.py` now read that same filename (the
`_parallel` suffix bug is fixed alongside the directory), from the same
`<dir>` — both are `GARGAML_RESULTS_DIR`, which `tree_synth.slurm` and the
synthetic measure jobs default to `results-revision/` under Slurm. A bare
`python scripts/...` run still defaults to `results/`. Populating
`results-revision/` with fresh synthetic stage-1 measures is still a separate,
not-yet-run step — nobody has submitted that array against the new directory
yet. To reproduce the old pre-`c5fba86` archived numbers, submit
`tree_synth.slurm` with `GARGAML_RESULTS_DIR=results-0` explicitly.

**`distribution_scores.py` no longer hardcodes `results-3/`.** It now reads
and writes through the same `GARGAML_RESULTS_DIR` every other script uses
(`results-revision/` under Slurm by default). To reproduce the old
pre-`c5fba86` archived directed base-score numbers, submit
`distribution_scores.slurm` with `GARGAML_RESULTS_DIR=results-3` explicitly.

**`LI-Large_nolouvain` (index 9) is expected to be infeasible, not slow.** The
reduction is what keeps a second-order ego graph small; without it HI-Small's
egos reach ~14,900 accounts and each is densified to roughly 1.8 GB. Record that
outcome — "what the pre-processing buys" is exactly what R2-M3 asks — rather
than resubmitting it indefinitely. `HI-Small_nolouvain` (index 8) is the arm
worth actually attempting, and it is far more expensive than the `_res<r>` arms.

**Two metrics files now reach a table.** `build_tables.py` used to pick up 8 of
the 33 `*_metrics.csv` on disk: excluding `*_diagnosis_metrics.csv` was
deliberate, but excluding the two `*_partial_observability_metrics.csv` was an
accident of the filename regex (they carry no `_directed`/`_undirected`
token). `src/utils/reporting.py`'s `TIDY_PATTERN` now has a direction-less
fallback (`NO_DIRECTION_PATTERN`), so those two files parse correctly and
task 5's appendix numbers reach a table like everything else.

**Pooled outputs are now guarded under sharding, not immune to it.**
`directed_diagnosis.py` and `pattern_splitting.py` each concatenate over their
whole dataset list into one fixed filename, so one-dataset-per-task would have
every task overwrite the pool with its own single-dataset version. Sharding
these is still possible — nothing stops `GARGAML_DATASET`/`GARGAML_DATASET_INDEX`/
`SLURM_ARRAY_TASK_ID` from being set — but both scripts now detect that case
and skip the pooled write with a one-line explanation instead of silently
corrupting it; the per-dataset files are written normally either way. Both are
still kept as single serial jobs here, which is the simpler way to get a
correct pooled file in one submission.

**`scripts/test_parallel.py` is not a sanity check** and is deliberately absent
from this harness. It runs the full 66-dataset directed sweep including the
100,000-node tier, and appends to the same `results/time_results_dir.txt` the
real measure scripts use — a concrete way to contaminate the timing log behind
the scalability figure.

---

## Choices made here, and how to change them

**Worker count is not auto-detected in stage 1.** All four measure scripts cap their pool
at `min(4, cpu_count() // 2)`, and that is left alone on purpose: runtime is a
*published result* here, so worker count and node sharing change reported
numbers, not just throughput. The consequence is arithmetic — `--cpus-per-task=8`
is what yields the intended 4 workers. Sixteen cores leave twelve idle, and
`--cpus-per-task=1` makes `Pool(processes=0)`, which raises. `GARGAML_N_CPU`
overrides it if you decide to change the published basis; log that decision.

`gargaml_tree.py` is the exception, and follows the allocation: its pool width
defaults to `SLURM_CPUS_PER_TASK` (else `min(4, cpu_count() // 2)`), and
`GARGAML_N_CPU=1` fits serially with no pool. No runtime from stage 2 is
reported, and its outputs are byte-identical for any width — every estimator is
seeded and single-threaded, and records are reassembled in grid order — so the
width buys throughput and nothing else. It matters: the serial grid is ~20 h of
single-core fitting on HI-Small, which is why job 62171623 hit its 16 h limit
with 19 of 20 cores idle. `tree.slurm` asks for 36, half a wICE thin node.

**Wall times are mostly headroom, not measurements.** The archived
`time_results_*.txt` files record only `<dataset>: <seconds>` with no job, host
or worker count, three scripts append to the same file, and `results-3`'s
timings cannot belong to its CSVs — so they support a median, not a bound. Each
header says which number is evidence and which is slack. New runs write one
timing file per task under `results/timing/`, with job id, host, worker count and
timestamp; `collect.slurm` concatenates them into `results/timing_all.csv`. The
legacy files are still appended to, so `VisualisationRunTime.ipynb` is unaffected.

**Fail-fast, so the mail tells the truth.** Every job runs `set -eo pipefail`
right after `source activate gargaml`. Before this, the closing `echo Finished`
set the job's exit code, so a `ModuleNotFoundError` or `FileNotFoundError` seconds
in still showed up as COMPLETED / ExitCode 0. Now the first failing command ends
the job as FAILED and the FAIL mail is sent. It is turned on *after* activation
because conda's activate scripts are not written for `-e`; the activation is
checked explicitly instead, against `CONDA_DEFAULT_ENV`. `-u` is left off for the
same reason. A Python script that decides to skip and returns normally still
exits 0 — for example `gargaml_tree.py` when no measures CSV exists for the
dataset, or a measure script whose output is already on disk.

**Not added, deliberately — tell me if you want them.**
`export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` is present but commented out in
`common.sh`. No disk-staging variant exists — it assumes `data/` is populated.
