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
| `GARGAML_N_CPU` | worker-pool width (see the caveat below) |
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
Drop the `--dependency` and gate on the files instead -- see Recovery.

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
ts=$(slurm/submit.sh   slurm/tree_synth.slurm base --dependency=afterok:$sd1:$su1)

# --- Stage 2: GraphSAGE (1 job, GPU) ---------------------------------------
# afterok on the TREE job: it consumes results/<dataset>_folds.csv.
gs=$(slurm/submit.sh slurm/graphsage.slurm HI-Small --dependency=afterok:$tree)

# --- Appendices / diagnostics (CPU, independent of stage 1) -----------------
ps=$(slurm/submit.sh  slurm/pattern_splitting.slurm HI-Small)
po1=$(slurm/submit.sh slurm/partial_obs.slurm 012)
po2=$(slurm/submit.sh slurm/partial_obs.slurm top50)
dd=$(sbatch --parsable slurm/directed_diagnosis.slurm | cut -d";" -f1)

# --- Reporting (1 core, minutes) -------------------------------------------
# afterANY, so one failed arm does not block the tables: build_tables.py
# announces missing tables and renders unfittable cells as "--" by design.
slurm/submit.sh slurm/collect.slurm all \
  --dependency=afterany:$tree:$blk:$ifj:$ts:$gs:$ps:$po1:$po2:$dd
```

LI-Large is the documented 16 h / 200 GB job and needs explicit overrides:

```bash
d7=$(slurm/submit.sh slurm/measures_ibm_dir.slurm   LI-Large --array=7 --time=16:00:00 --mem=200g)
u7=$(slurm/submit.sh slurm/measures_ibm_undir.slurm LI-Large --array=7 --time=16:00:00 --mem=200g)
slurm/submit.sh slurm/tree.slurm LI-Large --array=7 --time=16:00:00 --mem=200g \
  --dependency=afterok:$d7:$u7
```

---

## What depends on what

```
data/  ──┬─► measures_ibm_{dir,undir}   ──┬─► tree ──► graphsage   (folds.csv)
         │   (per dataset NAME)           ├─► tree_blocks
         │                                └─► if
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

Stage 2 and GraphSAGE do **not** resume at cell granularity — their metric
writers run after the loops, not inside them. GraphSAGE checkpoints every epoch
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

**Worker count is not auto-detected.** All four measure scripts cap their pool
at `min(4, cpu_count() // 2)`, and that is left alone on purpose: runtime is a
*published result* here, so worker count and node sharing change reported
numbers, not just throughput. The consequence is arithmetic — `--cpus-per-task=8`
is what yields the intended 4 workers. Sixteen cores leave twelve idle, and
`--cpus-per-task=1` makes `Pool(processes=0)`, which raises. `GARGAML_N_CPU`
overrides it if you decide to change the published basis; log that decision.

**Wall times are mostly headroom, not measurements.** The archived
`time_results_*.txt` files record only `<dataset>: <seconds>` with no job, host
or worker count, three scripts append to the same file, and `results-3`'s
timings cannot belong to its CSVs — so they support a median, not a bound. Each
header says which number is evidence and which is slack. New runs write one
timing file per task under `results/timing/`, with job id, host, worker count and
timestamp; `collect.slurm` concatenates them into `results/timing_all.csv`. The
legacy files are still appended to, so `VisualisationRunTime.ipynb` is unaffected.

**Not added, deliberately — tell me if you want them.** `set -euo pipefail` is
absent (without it a failed `source activate` still runs `python` and the job
reports success; with it a harmless nonzero return from `activate` aborts the
job). `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` is present but commented out
in `common.sh`. No disk-staging variant exists — it assumes `data/` is populated.
