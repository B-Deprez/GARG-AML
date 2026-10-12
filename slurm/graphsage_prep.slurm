#!/bin/bash
# GraphSAGE preparation, CPU only -- scripts/graphsage_baseline.py with
# GARGAML_PREPARE_ONLY=1.
#
# WHAT     Builds the three caches the GraphSAGE fits read -- the per-account
#          label table, the graph structure and both feature matrices -- and
#          exits without fitting. Writes $GARGAML_RESULTS_DIR/<dataset>_graphsage_
#          {labels.pkl,structure.pt,x_topology.pt,x_attributes.pt} and
#          <dataset>_graphsage_prep.csv, the cold build time and peak host
#          memory of each step.
# WHY      The label build is the baseline's host-memory peak and needs no GPU.
#          On LI-Large it peaked at 166.6 GiB MaxRSS (label_distribution.py's
#          identical call, job 62206918, 38 min), while wICE's gpu_a100
#          partition allows at most 126,000 MiB per GPU. So it runs here, and
#          graphsage.slurm only loads the result.
# SUBMIT   slurm/submit.sh slurm/graphsage_prep.slurm HI-Small
#          slurm/submit.sh slurm/graphsage_prep.slurm LI-Large --time=06:00:00 --mem=200g
# ARRAY    None. One dataset per invocation, like graphsage.slurm.
# PREREQ   None -- reads only data/. It does not need the folds file, so it can
#          run alongside stage 1. graphsage.slurm needs both this job's caches
#          and the tree job's <dataset>_folds.csv.
# RESUMES  Per cache file: a step whose file already exists is loaded, not
#          rebuilt, and flagged from_cache in the prep record. Nothing is ever
#          invalidated -- delete a cache file to rebuild it.
# CPUS     4 -- serial pandas passes, no pool.
# MEM      64g for HI-Small (the whole local run peaked at 9.1 GB).
#          LI-Large: --mem=200g, headroom over the measured 166.6 GiB label peak.
# TIME     02:00:00 for HI-Small, which takes minutes.
#          LI-Large: --time=06:00:00. Labels took 38 min in job 62206918; the
#          structure build and the attribute pass each read the 16.7 GB CSV
#          once more and are unmeasured.

#SBATCH --job-name=gargaml_graphsage_prep
#SBATCH --output=slurm/logs/gargaml_graphsage_prep_%j.out
#SBATCH --error=slurm/logs/gargaml_graphsage_prep_%j.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,BEGIN,END

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
export GARGAML_PREPARE_ONLY=1

echo "[$(date)] graphsage_prep: dataset=$GARGAML_DATASET configs=${GARGAML_CONFIGS:-default(topology,attributes)} results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/graphsage_baseline.py

echo Finished
