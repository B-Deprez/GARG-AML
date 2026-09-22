#!/bin/bash
# Base-score evaluation -- scripts/distribution_scores.py.
#
# WHAT     Evaluates the PURE GARG-AML score (no model fitted) against every
#          (cut-off, pattern) cell, with task 7's per-fold and pooled
#          out-of-fold breakdown on the IBM data. This is the base-score row of
#          Tables 10-11. Writes results/<dataset>_<direction>_base_metrics.csv,
#          which build_tables.py picks up.
# SUBMIT   sbatch slurm/distribution_scores.slurm
# ARRAY    None -- its dataset/direction loop lives inside `if __name__`, so the
#          env layer cannot reach it (see the NOTE).
# PREREQ   Stage-1 measures, and results/<dataset>_folds.csv if you want the
#          per-fold rows (a missing folds file is a normal state, not an error).
# RESUMES  NO.
# CPUS     4 -- serial, single-threaded.
# MEM      64g -- it parses the full transactions file once per pass (4 passes:
#          2 datasets x 2 directions) and evaluates 45 cells over 515,080 or
#          2,054,390 accounts. Raise to 200g if LI-Large stays in its list.
# TIME     06:00:00 -- no timing instrumentation exists in this script (its
#          `import timeit` is dead and it writes no time_results file), so this
#          is headroom, not a measured bound. The LI-Large arm is unmeasured.
#
# NOTE -- INPUT PATH, read before trusting the output.
#          This script reads results-3/<dataset>_GARGAML_<direction>.csv
#          hardcoded, NOT results/. Per CLAUDE.md, results-3 holds the pre-fix
#          (pre-c5fba86) directed measures and is byte-identical to results-0.
#          So its directed base-score numbers come from the archive, not from
#          what the stage-1 jobs write. Repoint or parameterise that path before
#          using this to report post-fix numbers.
#          Its dataset list is also inside `if __name__`, so GARGAML_DATASET
#          does not select anything here -- it runs HI-Small and LI-Large both.

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

echo "[$(date)] distribution_scores: datasets=HI-Small,LI-Large (script constants) reading=results-3/ (see header note)"

python -u scripts/distribution_scores.py

echo Finished
