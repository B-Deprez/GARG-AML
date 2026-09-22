#!/bin/bash
# Task 5 appendix, partial observability -- scripts/partial_observability.py.
#
# WHAT     Compares the PURE GARG-AML score on the full graph against the same
#          score on a single-bank view, for that institution's own clients,
#          with Louvain OFF on both sides -- otherwise the two regimes are
#          reduced by different partitions and the degradation cannot be
#          attributed to the missing edges. Its "full" column is therefore
#          deliberately NOT the published Tables 10-11 number.
# SUBMIT   slurm/submit.sh slurm/partial_obs.slurm top50
#          The positional arg names the institution, not a dataset:
#            slurm/submit.sh slurm/partial_obs.slurm 012
#            slurm/submit.sh slurm/partial_obs.slurm top50
#          It is exported as GARGAML_INSTITUTIONS, so one institution per job.
# ARRAY    None -- there are two institutions and they differ ~14x in size.
# PREREQ   None -- reads only data/. Independent of stage 1.
# RESUMES  NO.
# CPUS     8 -> the script's 4-worker pool (it is NOT serial: it uses the same
#          min(4, cpu_count()//2) pool as the measure scripts).
# MEM      180g. This is the memory-hungry job in the set and the reason is in
#          the script's own docstring: with Louvain off, second-order
#          neighbourhoods reach ~14,900 accounts, and the measure functions
#          densify each with nx.adjacency_matrix(...).toarray() -- about 1.8 GB
#          for a single node. Only the bank's clients are scored, which is what
#          keeps it finite. 180g is headroom over that per-node peak across 4
#          concurrent workers (4 x 1.8 GB is the floor, not the ceiling).
# TIME     12:00:00. Bank 012 (12,180 nodes) is far cheaper than top50 (164,822);
#          the two output files on disk are ~3 h apart, which is consistent with
#          top50 taking hours but is a file mtime, not a measurement. 12 h is
#          deliberate headroom for top50.

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

echo "[$(date)] partial_obs: institutions=$GARGAML_INSTITUTIONS dataset=${GARGAML_DATASET:-default(HI-Small)} louvain=off"

python -u scripts/partial_observability.py

echo Finished
