#!/bin/bash
# Task 6, directed-vs-undirected diagnosis -- scripts/directed_diagnosis.py.
#
# WHAT     Measures WHY the undirected score beats the directed one: a level
#          census, a reciprocal-edge census and seven score variants per node on
#          the synthetic grid, where pattern membership is ground truth.
# SUBMIT   slurm/submit.sh slurm/directed_diagnosis.slurm synthetic
# ARRAY    None by default. The 22 n=100 datasets already ran; widening to the
#          10,000-node tier is a GARGAML_DATASETS override.
# PREREQ   None -- reads data/ directly, and runs with Louvain OFF by default.
# RESUMES  NO.
# CPUS     4 -- serial, no pool.
# MEM      32g at the 100-node tier; raise to 64g if you widen to 10,000 nodes.
# TIME     06:00:00. The 22 n=100 datasets run in seconds; the 10,000-node tier
#          takes minutes per dataset. HI-Small is feasible only sampled, and a
#          sampled run deliberately writes NO metrics file (the censuses stay
#          honest, the ranking metrics would not).
#
# CAUTION  results/directed_diagnosis_{summary,metrics}.csv are pooled over the
#          whole DATASETS list, so a sharded run would have each task overwrite
#          the pool. Keep it as one job.

#SBATCH --job-name=gargaml_directed_diagnosis
#SBATCH --output=gargaml_directed_diagnosis.out
#SBATCH --error=gargaml_directed_diagnosis.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=32g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh
: # no per-dataset resolution: this job runs a grid, not one dataset
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] directed_diagnosis: datasets=${GARGAML_DATASETS:-default(22 x n=100 synthetic)} sample_nodes=${GARGAML_SAMPLE_NODES:-default(20000)} results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/directed_diagnosis.py

echo Finished
