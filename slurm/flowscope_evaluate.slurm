#!/bin/bash
# FlowScope's IBM metrics -- Flowscope_evaluate.py in the FlowScope fork.
#
# WHAT     Scores FlowScope's saved per-account scores against the GARG-AML
#          labels at every cut-off and target, with the same metrics module as
#          GARG-AML's own models (P@K, R@K, lift@K, TP@K, ties@K, AUC-ROC,
#          AUC-PR, precision, F1), plus per-fold rows and the pooled fold -1 on
#          GARG-AML's persisted CV partition. FlowScope itself is NOT re-run:
#          the script only reads <dataset>_map_id.pkl and <dataset>_map_scores.pkl.
#          Writes <dataset>_directed_flowscope_metrics.csv into the fork's
#          directory, and this job then copies it into $GARGAML_RESULTS_DIR so
#          scripts/build_tables.py and VisualisationResults.ipynb find it.
# SUBMIT   slurm/submit.sh slurm/flowscope_evaluate.slurm LI-Large
#          slurm/submit.sh slurm/flowscope_evaluate.slurm HI-Small
# PREREQ   <dataset>_folds.csv in $GARGAML_RESULTS_DIR (written by the tree job);
#          without it only the full-population rows are written, which the
#          script says in the log. The FlowScope fork on the VSC with its
#          <dataset>_map_id.pkl / _map_scores.pkl, and <dataset>_Trans.csv /
#          <dataset>_Patterns.txt either under its inputData/IBM/raw or in
#          GARG-AML's data/.
# ARRAY    None -- one dataset per submission.
# RESUMES  No; the output is overwritten.
# CPUS     2 -- serial pandas, no multiprocessing.
# MEM      200g. The cost is define_ML_labels reading the 16.7 GB
#          LI-Large_Trans.csv into pandas -- the same step as
#          label_distribution.slurm, whose LI-Large run peaked at 166.6 GiB
#          (job 62206918). 200g leaves ~20% headroom; the 54 cell evaluations
#          add a few GB on top. HI-Small needs far less: pass --mem=64g.
# TIME     06:00:00. Label construction took 38 min in label_distribution's
#          LI-Large run; the 54 cell evaluations (each up to 2.05M accounts, 5
#          folds + pooled) are the unmeasured part, so the budget is generous.
#
# Where the fork lives is not recorded in this repository: FLOWSCOPE_DIR below
# is a guess (a sibling of GARG-AML) -- export FLOWSCOPE_DIR before submitting
# if it is elsewhere.

#SBATCH --job-name=gargaml_flowscope_eval
#SBATCH --output=slurm/logs/gargaml_flowscope_eval.out
#SBATCH --error=slurm/logs/gargaml_flowscope_eval.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --mem=200g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML || exit 1
GARGAML_ROOT="$PWD"

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
RESULTS_ABS="$GARGAML_ROOT/$GARGAML_RESULTS_DIR"

# The script only understands the two IBM datasets; a view tag would silently
# be rejected by argparse after the job has already queued.
case "$GARGAML_DATASET" in
  HI-Small|LI-Large) ;;
  *) echo "ERROR: Flowscope_evaluate.py supports HI-Small and LI-Large, not '$GARGAML_DATASET'." >&2; exit 1 ;;
esac

FLOWSCOPE_DIR="${FLOWSCOPE_DIR:-$(dirname "$GARGAML_ROOT")/FlowScope/flowscope}"
[ -f "$FLOWSCOPE_DIR/Flowscope_evaluate.py" ] || { echo "ERROR: no Flowscope_evaluate.py in $FLOWSCOPE_DIR (set FLOWSCOPE_DIR)" >&2; exit 1; }

# Raw transactions: the fork's own copy if it has one, else GARG-AML's data/.
if [ -f "$FLOWSCOPE_DIR/inputData/IBM/raw/${GARGAML_DATASET}_Trans.csv" ]; then
  DATA_DIR="$FLOWSCOPE_DIR/inputData/IBM/raw"
else
  DATA_DIR="$GARGAML_ROOT/data"
fi

for f in "${GARGAML_DATASET}_map_id.pkl" "${GARGAML_DATASET}_map_scores.pkl"; do
  [ -f "$FLOWSCOPE_DIR/$f" ] || { echo "ERROR: $FLOWSCOPE_DIR/$f missing -- run FlowScope_ibm.py first" >&2; exit 1; }
done
for f in "${GARGAML_DATASET}_Trans.csv" "${GARGAML_DATASET}_Patterns.txt"; do
  [ -f "$DATA_DIR/$f" ] || { echo "ERROR: $DATA_DIR/$f missing" >&2; exit 1; }
done
[ -f "$RESULTS_ABS/${GARGAML_DATASET}_folds.csv" ] || \
  echo "WARNING: $RESULTS_ABS/${GARGAML_DATASET}_folds.csv missing -- only full-population rows will be written." >&2

echo "[$(date)] flowscope_evaluate: dataset=$GARGAML_DATASET fork=$FLOWSCOPE_DIR data=$DATA_DIR folds=$RESULTS_ABS"

# Run from the fork: the script loads its pickles and imports utils/ relative to cwd.
cd "$FLOWSCOPE_DIR"
python -u Flowscope_evaluate.py --datasets "$GARGAML_DATASET" \
  --data-dir "$DATA_DIR" --folds-dir "$RESULTS_ABS" --out-dir "$FLOWSCOPE_DIR"

OUT="${GARGAML_DATASET}_directed_flowscope_metrics.csv"
[ -s "$FLOWSCOPE_DIR/$OUT" ] || { echo "ERROR: $OUT was not written" >&2; exit 1; }
cp "$FLOWSCOPE_DIR/$OUT" "$RESULTS_ABS/$OUT"
echo "copied $OUT -> $RESULTS_ABS"

echo Finished
