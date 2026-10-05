#!/bin/bash
# Base-score evaluation -- scripts/distribution_scores.py.
#
# WHAT     Evaluates the pure GARG-AML score (no model fitted) against every
#          (cut-off, pattern) cell, with the per-fold and pooled out-of-fold
#          breakdown on the IBM data. This is the base-score row of the result
#          tables. Writes <dataset>_<direction>_base_metrics.csv, which
#          build_tables.py picks up; on the synthetic grid,
#          synthetic_score_<direction>_supervised.csv (both directions, all
#          66 datasets), which notebooks/VisualisationResults.ipynb reads.
# SUBMIT   The optional argument becomes GARGAML_DATASET:
#            sbatch slurm/distribution_scores.slurm                     # HI-Small and LI-Large
#            slurm/submit.sh slurm/distribution_scores.slurm HI-Small   # one IBM dataset
#            slurm/submit.sh slurm/distribution_scores.slurm synthetic  # the synthetic grid
# ARRAY    None -- one job loops both directions over what it selected.
# PREREQ   Stage-1 measures, and <dataset>_folds.csv if you want the per-fold
#          rows (a missing folds file is a normal state, not an error).
# RESUMES  No.
# CPUS     4 -- serial, single-threaded.
# MEM      64g -- it parses the full transactions file once per pass (4 passes:
#          2 datasets x 2 directions) and evaluates 45 cells over 515,080 or
#          2,054,390 accounts. Raise to 200g while LI-Large is in its list.
# TIME     06:00:00 -- the script carries no timing instrumentation and writes no
#          time_results file, so this is headroom rather than a measured bound.
#          The LI-Large arm is unmeasured. The synthetic grid fits easily in a
#          fraction of that: it reads measure files and labels, no transactions.
#
# NOTE     It reads the measures from $GARGAML_RESULTS_DIR, the same directory
#          every other script uses; a bare `python scripts/...` run defaults to
#          results/.

#SBATCH --job-name=gargaml_dist_scores
#SBATCH --output=slurm/logs/gargaml_dist_scores.out
#SBATCH --error=slurm/logs/gargaml_dist_scores.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML || exit 1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml
# Fail fast from here on, so a crash ends the job as FAILED (FAIL mail) rather
# than COMPLETED via the closing echo. Set after activation, which is not -e
# safe; the activation is checked explicitly instead.
[ "${CONDA_DEFAULT_ENV:-}" = "gargaml" ] || { echo "ERROR: conda env gargaml did not activate" >&2; exit 1; }
set -eo pipefail

source slurm/common.sh
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

if [ -n "${1:-}" ]; then
  export GARGAML_DATASET="$1"
fi

echo "[$(date)] distribution_scores: datasets=${GARGAML_DATASET:-HI-Small,LI-Large (script default)} reading=$GARGAML_RESULTS_DIR/ results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/distribution_scores.py

echo Finished
