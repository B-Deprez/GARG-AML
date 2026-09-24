#!/bin/bash
# Stage 2, unsupervised isolation forest -- scripts/gargaml_IF.py.
#
# WHAT     Fits an isolation forest on the GARG-AML feature table and evaluates
#          it against the same labels, as the unsupervised comparison.
# SUBMIT   sbatch slurm/if.slurm
# ARRAY    None.
# PREREQ   The same stage-1 measures as slurm/tree.slurm, same dataset name.
# RESUMES  No.
# CPUS     4 -- IsolationForest is fit once per cell, not in parallel here.
# MEM      64g, the same feature-table reasoning as tree.slurm.
# TIME     02:00:00 -- an isolation forest is cheaper than the boosting grid.
#
# NOTE     gargaml_IF.py holds its dataset/direction choice inside main(), not
#          at module scope (unlike gargaml_tree.py, gargaml_directed.py, ...),
#          so there is no env var to select a dataset here. Edit the script's
#          own `dataset = "HI-Small"` constant to run a different one.

#SBATCH --job-name=gargaml_if
#SBATCH --output=gargaml_if.out
#SBATCH --error=gargaml_if.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] if: running scripts/gargaml_IF.py's own dataset/direction constants results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/gargaml_IF.py

echo Finished
