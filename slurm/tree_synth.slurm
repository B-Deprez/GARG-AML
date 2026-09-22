#!/bin/bash
# Stage 2 on the synthetic grid -- scripts/gargaml_tree_synthetic{,_3,_5}.py.
#
# WHAT     Fits the tree and boosting models across the 66 synthetic datasets
#          and writes the per-variant tables that feed the average-rank and
#          critical-difference analysis (Friedman + Nemenyi, k=8, N=66).
# SUBMIT   The positional argument selects the variant:
#            slurm/submit.sh slurm/tree_synth.slurm base   # gargaml_tree_synthetic.py
#            slurm/submit.sh slurm/tree_synth.slurm 3      # ..._3.py  (3 injected patterns)
#            slurm/submit.sh slurm/tree_synth.slurm 5      # ..._5.py  (5 injected patterns)
# ARRAY    None -- each script loops the whole grid internally and the fits are
#          seconds apiece. The expensive part of the synthetic work is stage 1,
#          which IS an array job.
# PREREQ   The synthetic stage-1 measures. READ THE WARNING BELOW FIRST.
# RESUMES  NO.
# CPUS     4 -- serial sklearn fits, cores for the pandas passes.
# MEM      32g -- 66 datasets are read one at a time, and the largest is a
#          100,000-node measure frame (~18 columns), not a transaction file.
# TIME     02:00:00 -- 66 datasets x a handful of cheap fits each.
#
# WARNING -- INPUT PATH MISMATCH, read before submitting.
#          These three scripts read  results-0/<dataset>_GARGAML_<direction>.csv
#          while the stage-1 synthetic jobs write
#                                    results/<dataset>_GARGAML_directed.csv  and
#                                    results/<dataset>_GARGAML_undirected_parallel.csv
#          So a fresh stage-1 run does NOT feed this job: the directory differs
#          for both directions, and the filename differs for undirected.
#          Today this job therefore reads the ARCHIVED results-0 measures, which
#          per CLAUDE.md are pre-c5fba86 for the directed arm. Resolve the
#          naming before trusting a rerun -- see slurm/README.md, "Known gap:
#          synthetic stage 1 -> stage 2". This job is left submittable because
#          it reproduces today's published numbers from today's archive; it is
#          not wired to the new measures.

#SBATCH --job-name=gargaml_tree_synth
#SBATCH --output=gargaml_tree_synth.out
#SBATCH --error=gargaml_tree_synth.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh

case "${1:-base}" in
  base) SYNTH_SCRIPT="scripts/gargaml_tree_synthetic.py" ;;
  3)    SYNTH_SCRIPT="scripts/gargaml_tree_synthetic_3.py" ;;
  5)    SYNTH_SCRIPT="scripts/gargaml_tree_synthetic_5.py" ;;
  *)    echo "ERROR: variant must be base, 3 or 5 (got '${1}')" >&2; exit 1 ;;
esac

echo "[$(date)] tree_synth: variant=${1:-base} script=$SYNTH_SCRIPT reading=results-0/ (see header warning)"

python -u "$SYNTH_SCRIPT"

echo Finished
