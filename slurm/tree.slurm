#!/bin/bash
# Stage 2, decision tree + gradient boosting -- scripts/gargaml_tree.py.
#
# WHAT     Reads the stage-1 block measures back, builds the feature table
#          (score + neighbourhood summaries), and fits/evaluates the tree and
#          boosting models over the four feature configs x both directions x
#          cut-offs x targets x folds. Writes the
#          <dataset>_<metric>_<model>_<direction>_combined.csv matrices, the
#          tidy <dataset>_<direction>_metrics.csv, and -- with two or more folds
#          -- <dataset>_folds.csv, which the GraphSAGE job consumes.
# SUBMIT   sbatch --array=4 slurm/tree.slurm
#          slurm/submit.sh slurm/tree.slurm HI-Small --array=4
# ARRAY    --array=<i> indexes slurm/common.sh's IBM_DATASETS (same map as
#          stage 1, so array index 4 is HI-Small in both).
# PREREQ   The matching stage-1 jobs for the same dataset name:
#            dir=$(slurm/submit.sh slurm/measures_ibm_dir.slurm HI-Small --array=4)
#            und=$(slurm/submit.sh slurm/measures_ibm_undir.slurm HI-Small --array=4)
#            slurm/submit.sh slurm/tree.slurm HI-Small --array=4 \
#              --dependency=afterok:$dir:$und
#          A missing measures CSV is not fatal: gargaml_tree.py keeps the
#          directions whose file exists and reports the gap, so afterany is also
#          defensible if the undirected arm should proceed when the directed one
#          fails.
# RESUMES  Per section, not per cell, and not automatically. Each of the 7
#          direction/feature-config sections writes its files when it finishes
#          (run_config ends in write_metrics), so a wall-time kill loses only
#          the section in progress -- job 62171623 kept six and lost the
#          seventh. There is no skip-if-done: a resubmission refits every
#          section and rewrites <dataset>_folds.csv (identically; the partition
#          is seeded). Keep the wall time generous rather than relying on a
#          restart.
# CPUS     36, half a 72-core wICE thin node. gargaml_tree.py fits in a spawn
#          process pool, one (cut-off, target, fold) split per task and both
#          models per split, whose width follows SLURM_CPUS_PER_TASK -- so this
#          line is the knob. Every estimator is seeded and single-threaded (BLAS
#          limited to 1) and the parent reassembles records in grid order, so
#          the outputs are byte-identical for any width. Stage 1 keeps a fixed
#          cap because its runtime is a reported result (Fig. 6); no runtime
#          from this stage is reported. The serial loop this replaced left 19 of
#          job 62171623's 20 cores idle for the whole run.
# MEM      64g for HI-Small. The parent holds the transaction frame, the reduced
#          graph and the feature table (515,080 rows x ~40 float64 columns,
#          ~165 MB), and the sweep keeps per-cell score arrays. Each worker
#          holds its own copy of one section's feature matrix and the label
#          columns, one training slice and sklearn's float32 copy of it: an
#          estimated 0.5-1 GB (0.6 GB measured for the 18-feature directed
#          blocks config in a 2-fold test), so ~20-35 GB for 36 workers. Headroom over that,
#          not a measured peak. A worker killed for memory fails the job at once
#          (BrokenProcessPool) rather than hanging it to the wall time.
#          LI-Large: --cpus-per-task=36 --time=16:00:00 --mem=200g (an estimated
#          1.5-2 GB per worker, ~55-70 GB for 36).
# TIME     06:00:00 for HI-Small. Serially the grid could not finish in 16 h:
#          job 62151480 (2026-09-25) was killed by a 06:00:00 limit 3.2 of 7
#          sections in, and job 62171623 (2026-09-28) by the 16:00:00 limit ~19
#          of 46 fitted cells into the 7th and last (directed / all, 28
#          features). Fitting both runs to a cost of cells x feature count gives
#          ~16.5 s per cell-feature, i.e. ~20 h of single-core fitting for the
#          whole grid. Estimated on 36 workers: 1-1.5 h of fitting plus data
#          preparation (graph build, Louvain, labels; unmeasured, well under an
#          hour), so 6 h is a wide margin.
#          LI-Large keeps 16 h. 4x the rows at headline cut-offs only
#          (DATASET_SETTINGS) is an estimated 60-80 h single-core, roughly 2-4 h
#          on 36 workers, but its data preparation -- the 16 GB CSV read
#          twice, the 176M-edge graph and Louvain -- is unmeasured and likely
#          dominates.
#          After each section the log prints "fitted <n> splits on <w> workers
#          in <t> min; peak worker memory <m> GB"; replace these estimates with
#          those lines from the first parallel run.
#
# KNOBS    GARGAML_N_FOLDS picks the split: 0 is a single 70/30 split, 2 or more
#          is that many stratified folds. Use 2 when checking a code change
#          rather than producing paper numbers. It also rewrites
#          <dataset>_folds.csv, which GraphSAGE then inherits. CUT_OFFS and
#          TARGET_COLUMNS have no env override -- trim those two module
#          constants by hand.
#          GARGAML_N_CPU overrides the pool width (default SLURM_CPUS_PER_TASK,
#          else min(4, cpu_count() // 2) outside Slurm); 1 fits every split
#          serially in the parent, with no pool. Results do not depend on it.

#SBATCH --job-name=gargaml_tree
#SBATCH --output=slurm/logs/gargaml_tree_%A_%a.out
#SBATCH --error=slurm/logs/gargaml_tree_%A_%a.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --array=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=06:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh
resolve_dataset "${1:-}"
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] tree: dataset=$GARGAML_DATASET n_folds=${GARGAML_N_FOLDS:-default(5)} results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/gargaml_tree.py

echo Finished
