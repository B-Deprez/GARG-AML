#!/bin/bash
# Tables 7-8's label-percentage matrices -- scripts/label_distribution.py.
#
# WHAT     Percentage of accounts labelled 1 at each cut-off in CUT_OFFS
#          (including 0.0), per pattern type. This is the model- and
#          direction-independent `imbalance` metric, recomputed straight from
#          src/data/pattern_construction.py rather than through
#          gargaml_tree.py's data_preparation() -- no GARG-AML scores, no
#          Louvain reduction, no model fit. Writes
#          <dataset>_imbalance_{undirected,directed}_combined.csv (identical
#          content under both direction tokens), which
#          notebooks/DistributionScores.ipynb already reads to render Tables
#          7-8 (cells 29-32: read the CSV, x100, `.T.to_latex()`).
# SUBMIT   slurm/submit.sh slurm/label_distribution.slurm HI-Small
#          slurm/submit.sh slurm/label_distribution.slurm LI-Large
# ARRAY    None -- one dataset per submission (see resolve_dataset below); the
#          two only share the job script, not a run.
# PREREQ   None -- reads only data/. Independent of stage 1.
# RESUMES  No, but HI-Small is seconds and worth just rerunning.
# CPUS     4 -- serial, single-threaded (pandas passes only, no
#          multiprocessing.Pool and no model fit).
# MEM      64g. Headroom over one full read of <dataset>_Trans.csv into
#          pandas, one merge against the patterns file, and a groupby -- no
#          GARG-AML measures or fold file are read alongside it. Unmeasured
#          for LI-Large; raise to 200g if it OOMs, matching
#          distribution_scores.slurm and pattern_splitting.slurm's own
#          LI-Large headroom.
# TIME     04:00:00. HI-Small finishes in well under a minute locally; the
#          LI-Large arm is unmeasured -- define_ML_labels's per-row
#          Amount-formatting pass is the likely cost, since it is the same
#          step distribution_scores.py already pays for LI-Large under a 6h
#          budget. Raise if it does not finish.

#SBATCH --job-name=gargaml_label_distribution
#SBATCH --output=slurm/logs/gargaml_label_distribution.out
#SBATCH --error=slurm/logs/gargaml_label_distribution.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
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
resolve_dataset "${1:-}"
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] label_distribution: dataset=$GARGAML_DATASET results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/label_distribution.py

echo Finished
