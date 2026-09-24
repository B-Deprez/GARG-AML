#!/bin/bash
# Base-score evaluation -- scripts/distribution_scores.py.
#
# WHAT     Evaluates the pure GARG-AML score (no model fitted) against every
#          (cut-off, pattern) cell, with the per-fold and pooled out-of-fold
#          breakdown on the IBM data. This is the base-score row of the result
#          tables. Writes <dataset>_<direction>_base_metrics.csv, which
#          build_tables.py picks up.
# SUBMIT   sbatch slurm/distribution_scores.slurm
# ARRAY    None -- its dataset/direction loop lives inside `if __name__`, so the
#          env layer cannot reach it (see the note).
# PREREQ   Stage-1 measures, and <dataset>_folds.csv if you want the per-fold
#          rows (a missing folds file is a normal state, not an error).
# RESUMES  No.
# CPUS     4 -- serial, single-threaded.
# MEM      64g -- it parses the full transactions file once per pass (4 passes:
#          2 datasets x 2 directions) and evaluates 45 cells over 515,080 or
#          2,054,390 accounts. Raise to 200g while LI-Large is in its list.
# TIME     06:00:00 -- the script carries no timing instrumentation and writes no
#          time_results file, so this is headroom rather than a measured bound.
#          The LI-Large arm is unmeasured.
#
# NOTE     It reads the measures from $GARGAML_RESULTS_DIR, the same directory
#          every other script uses; a bare `python scripts/...` run defaults to
#          results/. Its dataset list is inside `if __name__`, so GARGAML_DATASET
#          selects nothing here -- it runs HI-Small and LI-Large both.

#SBATCH --job-name=gargaml_dist_scores
#SBATCH --output=gargaml_dist_scores.out
#SBATCH --error=gargaml_dist_scores.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] distribution_scores: datasets=HI-Small,LI-Large (script constants) reading=$GARGAML_RESULTS_DIR/ results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/distribution_scores.py

echo Finished
