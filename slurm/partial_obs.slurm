#!/bin/bash
# Partial observability -- scripts/partial_observability.py.
#
# WHAT     Compares the pure GARG-AML score on the full graph against the same
#          score on a single-bank view, for that institution's own clients,
#          with Louvain off on both sides -- otherwise the two regimes are
#          reduced by different partitions and the degradation cannot be
#          attributed to the missing edges. Its "full" column is therefore not
#          the main-table number, which is computed with Louvain on.
# SUBMIT   slurm/submit.sh slurm/partial_obs.slurm top50
#          The positional arg names the institution, not a dataset:
#            slurm/submit.sh slurm/partial_obs.slurm 012
#            slurm/submit.sh slurm/partial_obs.slurm top50
#          It is exported as GARGAML_INSTITUTIONS, so one institution per job.
# ARRAY    None -- there are two institutions and they differ ~14x in size.
# PREREQ   None -- reads only data/. Independent of stage 1.
# RESUMES  No.
# CPUS     8 -> the script's 4-worker pool (it is not serial: it uses the same
#          min(4, cpu_count()//2) pool as the measure scripts).
# MEM      180g, the largest request in the set, for the reason in the script's
#          own docstring: with Louvain off, second-order neighbourhoods reach
#          ~14,900 accounts, and the measure functions densify each with
#          nx.adjacency_matrix(...).toarray() -- about 1.8 GB for a single node.
#          Scoring only the bank's clients is what keeps it finite. 180g is
#          headroom over that per-node peak across 4 concurrent workers
#          (4 x 1.8 GB is the floor, not the ceiling).
# TIME     12:00:00. Bank 012 (12,180 nodes) is far cheaper than top50 (164,822
#          nodes), which takes hours; 12 h is deliberate headroom for top50.

#SBATCH --job-name=gargaml_partial_obs
#SBATCH --output=gargaml_partial_obs.out
#SBATCH --error=gargaml_partial_obs.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=180g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh
export GARGAML_INSTITUTIONS="${1:-012}"
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] partial_obs: institutions=$GARGAML_INSTITUTIONS dataset=${GARGAML_DATASET:-default(HI-Small)} louvain=off results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/partial_observability.py

echo Finished
