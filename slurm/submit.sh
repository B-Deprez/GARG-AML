#!/bin/bash
# Submit one GARG-AML job with a job name that identifies it in Slurm's mails.
#
#   slurm/submit.sh <script.slurm> <Dataset> [extra sbatch args...]
#
#   slurm/submit.sh slurm/measures_ibm_dir.slurm HI-Small
#   slurm/submit.sh slurm/tree.slurm LI-Large --time=16:00:00 --mem=200g
#   slurm/submit.sh slurm/measures_synth_dir.slurm synth --array=0-21 --time=00:30:00
#
# The dataset is passed through BOTH as --job-name and as the script's $1, so
# "squeue" and the FAIL mail say which arm failed rather than just which stage.
# Prints the job id on stdout (sbatch --parsable), so it chains:
#
#   first=$(slurm/submit.sh slurm/tree.slurm HI-Small)
#   slurm/submit.sh slurm/graphsage.slurm HI-Small --dependency=afterok:$first

if [ "$#" -lt 2 ]; then
  echo "usage: slurm/submit.sh <script.slurm> <Dataset> [extra sbatch args...]" >&2
  exit 1
fi

script="$1"; shift
dataset="$1"; shift

if [ ! -f "$script" ]; then
  echo "ERROR: no such job script: $script" >&2
  exit 1
fi

name="$(basename "$script" .slurm)_${dataset}"

# Extra sbatch args come BEFORE the script so they override the #SBATCH lines;
# the dataset goes after it, as the script's positional $1.
sbatch --parsable --job-name="$name" "$@" "$script" "$dataset"
