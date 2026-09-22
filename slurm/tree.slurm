#!/bin/bash
# Stage 2, decision tree + gradient boosting -- scripts/gargaml_tree.py.
#
# WHAT     Reads the stage-1 block measures back, builds the feature table
#          (score + neighbourhood summaries), and fits/evaluates the tree and
#          boosting models over the four task-3 feature configs x both
#          directions x cut-offs x targets x folds. Writes the legacy
#          <dataset>_<metric>_<model>_<direction>_combined.csv matrices, the
#          tidy <dataset>_<direction>_metrics.csv, and -- when N_FOLDS >= 2 --
#          results/<dataset>_folds.csv, which the GraphSAGE job consumes.
# SUBMIT   sbatch --array=4 slurm/tree.slurm
#          slurm/submit.sh slurm/tree.slurm HI-Small --array=4
# ARRAY    --array=<i> indexes slurm/common.sh's IBM_DATASETS (same map as
#          stage 1, so array index 4 is HI-Small in both).
# PREREQ   The matching stage-1 job for the SAME dataset name:
#            dir=$(slurm/submit.sh slurm/measures_ibm_dir.slurm HI-Small --array=4)
#            und=$(slurm/submit.sh slurm/measures_ibm_undir.slurm HI-Small --array=4)
#            slurm/submit.sh slurm/tree.slurm HI-Small --array=4 \
#              --dependency=afterok:$dir:$und
#          A missing measures CSV is NOT fatal: gargaml_tree.py:183 keeps only
#          the directions whose file exists and reports the gap. So afterany is
#          also defensible if you want the undirected arm to proceed when the
#          directed one fails.
# RESUMES  NO. A wall-time kill loses the whole grid for this dataset -- the
#          metric writers run after the loops, not inside them. Keep the wall
#          time generous rather than relying on a restart.
# CPUS     8. sklearn's tree/boosting fits here are single-threaded; the cores
#          are for pandas and the feature build, not for parallel fits.
# MEM      64g for HI-Small. The feature table is 515,080 rows x ~40 columns of
#          float64 (~165 MB) but it is built from the transaction frame and the
#          reduced graph held simultaneously, and the sweep holds per-cell
#          score arrays. 64g is headroom over that, not a measured peak.
#          LI-Large: --time=16:00:00 --mem=200g, the documented envelope.
# TIME     06:00:00 for HI-Small. A real HI-Small 5-fold run has been observed
#          to take tens of minutes to hours depending on the sweep, so six hours
#          is the safe default; LI-Large gets the documented 16 h.
#
# KNOBS    GARGAML_N_FOLDS=2 is the cheap verification setting (CLAUDE.md s2) --
#          use it when testing a code change rather than producing paper
#          numbers. Note it also rewrites results/<dataset>_folds.csv, which
#          GraphSAGE then inherits. CUT_OFFS/TARGET_COLUMNS have no env
#          override -- trim those two module constants by hand, as CLAUDE.md's
#          own testing guidance already does.

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

echo "[$(date)] tree: dataset=$GARGAML_DATASET n_folds=${GARGAML_N_FOLDS:-default(5)}"

python -u scripts/gargaml_tree.py

echo Finished
