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
# RESUMES  No. A wall-time kill loses the whole grid for this dataset -- the
#          metric writers run after the loops, not inside them. Keep the wall
#          time generous rather than relying on a restart.
# CPUS     8. The sklearn tree and boosting fits are single-threaded; the cores
#          are for pandas and the feature build, not for parallel fits.
# MEM      64g for HI-Small. The feature table is 515,080 rows x ~40 float64
#          columns (~165 MB), but it is built while the transaction frame and the
#          reduced graph are held, and the sweep keeps per-cell score arrays.
#          Headroom over that, not a measured peak.
#          LI-Large: --time=16:00:00 --mem=200g.
# TIME     06:00:00 for HI-Small, where a 5-fold run takes tens of minutes to
#          hours depending on the sweep; LI-Large gets 16 h.
#
# KNOBS    GARGAML_N_FOLDS picks the split: 0 is a single 70/30 split, 2 or more
#          is that many stratified folds. Use 2 when checking a code change
#          rather than producing paper numbers. It also rewrites
#          <dataset>_folds.csv, which GraphSAGE then inherits. CUT_OFFS and
#          TARGET_COLUMNS have no env override -- trim those two module
#          constants by hand.

#SBATCH --job-name=gargaml_tree
#SBATCH --output=gargaml_tree_%A_%a.out
#SBATCH --error=gargaml_tree_%A_%a.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --array=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
