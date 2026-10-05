#!/bin/bash
# Stage 2 on the synthetic grid -- scripts/gargaml_tree_synthetic{,_3,_5}.py.
#
# WHAT     Fits the tree and boosting models across the 66 synthetic datasets
#          and writes the per-variant tables that feed the average-rank and
#          critical-difference analysis (Friedman + Nemenyi, k=8, N=66).
# SUBMIT   The positional argument is <variant>_<direction>. The variant picks
#          the script, the direction the measures (GARGAML_DIRECTED):
#            slurm/submit.sh slurm/tree_synth.slurm 3_undirected  # ..._3.py  (3 injected patterns)
#            slurm/submit.sh slurm/tree_synth.slurm 3_directed
#            slurm/submit.sh slurm/tree_synth.slurm 5_undirected  # ..._5.py  (5 injected patterns)
#            slurm/submit.sh slurm/tree_synth.slurm 5_directed
#            slurm/submit.sh slurm/tree_synth.slurm base_directed # gargaml_tree_synthetic.py (3 and 5)
#          The four 3/5 runs are what notebooks/VisualisationResults.ipynb
#          reads. There is no default direction: the scripts used to hardcode
#          undirected, and a run that silently picks one is the easy mistake.
# WRITES   $GARGAML_RESULTS_DIR/synthetic_tree_<True|False>_<3|5|full>.csv.
# ARRAY    None -- each script loops the whole grid internally and the fits are
#          seconds apiece. The expensive part of the synthetic work is stage 1,
#          which is an array job.
# PREREQ   The synthetic stage-1 measures, in the directory this job reads:
#          <dir>/<dataset>_GARGAML_directed.csv and
#          <dir>/<dataset>_GARGAML_undirected_parallel.csv, where <dir> is
#          $GARGAML_RESULTS_DIR -- the same directory the measure jobs wrote to.
#          A bare `python scripts/...` run defaults to results/ instead.
# RESUMES  No.
# CPUS     4 -- serial sklearn fits, cores for the pandas passes.
# MEM      32g -- the 66 datasets are read one at a time, and the largest is a
#          100,000-node measure frame (~18 columns), not a transaction file.
# TIME     02:00:00 -- 33 datasets per 3/5 run (66 for base). Each dataset is
#          prepared once (graph, Louvain, neighbour summaries: ~160 s on the
#          densest 100,000-node datasets, 23 s on the sparsest, measured on a
#          laptop) and then fits 8 cheap models.

#SBATCH --job-name=gargaml_tree_synth
#SBATCH --output=slurm/logs/gargaml_tree_synth_%j.out
#SBATCH --error=slurm/logs/gargaml_tree_synth_%j.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32g
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

SPEC="${1:-}"
VARIANT="${SPEC%%_*}"
DIRECTION="${SPEC#*_}"

case "$VARIANT" in
  base) SYNTH_SCRIPT="scripts/gargaml_tree_synthetic.py" ;;
  3)    SYNTH_SCRIPT="scripts/gargaml_tree_synthetic_3.py" ;;
  5)    SYNTH_SCRIPT="scripts/gargaml_tree_synthetic_5.py" ;;
  *)    echo "ERROR: argument must be <base|3|5>_<directed|undirected> (got '${SPEC}')" >&2; exit 1 ;;
esac

case "$DIRECTION" in
  directed)   export GARGAML_DIRECTED=1 ;;
  undirected) export GARGAML_DIRECTED=0 ;;
  *)          echo "ERROR: argument must be <base|3|5>_<directed|undirected> (got '${SPEC}')" >&2; exit 1 ;;
esac

echo "[$(date)] tree_synth: variant=$VARIANT direction=$DIRECTION script=$SYNTH_SCRIPT reading=$GARGAML_RESULTS_DIR/ results_dir=$GARGAML_RESULTS_DIR"

python -u "$SYNTH_SCRIPT"

echo Finished
