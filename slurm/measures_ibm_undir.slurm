#!/bin/bash
# Stage 1, IBM undirected block measures -- scripts/gargaml_undirected.py.
#
# WHAT     Builds the transaction graph, applies the Louvain reduction the
#          dataset name asks for, and computes the three undirected block measures
#          per node. Writes results/<dataset>_GARGAML_undirected.csv.
# SUBMIT   sbatch --array=4 slurm/measures_ibm_undir.slurm
#          slurm/submit.sh slurm/measures_ibm_undir.slurm HI-Small --array=4
# ARRAY    --array=<i> indexes slurm/common.sh's IBM_DATASETS; index 4 is the
#          published HI-Small setting. Submit only the arms you want:
#            --array=0,1     task-5 bank views      (cheap)
#            --array=2,3,5,6 task-4 Louvain sweep   (~5 min each)
#            --array=4       HI-Small, published    (~5 min)
#            --array=7       LI-Large               NEEDS the overrides below
#            --array=8       HI-Small_nolouvain     NEEDS the overrides below
#            --array=9       LI-Large_nolouvain     expected INFEASIBLE, see README
# PREREQ   None -- stage 1 reads only data/. This is the head of the chain.
# RESUMES  YES, at dataset granularity. The script skips a dataset whose output
#          CSV already exists, so resubmitting the same array IS the recovery
#          procedure. Writes are atomic (temp + os.replace), so a wall-time kill
#          cannot leave a truncated file that the skip would mistake for done.
#          GARGAML_FORCE=1 recomputes anyway -- needed for the pending
#          regeneration of the directed measures after commit c5fba86.
# CPUS     8, sized to the code, not to the node. The script caps its pool at
#          min(4, cpu_count()//2), so 8 cores is exactly what yields its 4
#          workers; 16 would leave 12 idle, and --cpus-per-task=1 makes
#          Pool(processes=0), which raises. Worker count is NOT auto-detected on
#          purpose: it changes the runtimes the scalability figure reports.
# MEM      64g for the HI-Small arms. Basis: HI-Small is 515,080 nodes /
#          5,078,345 edges against LI-Large's 176,066,557 -- ~35x smaller on
#          edges than the documented 200 GB LI-Large envelope. construct_IBM_graph
#          reads the whole 475 MB CSV with pd.read_csv (no chunksize), and the
#          directed path then holds G, G_reduced, G_reduced_und and
#          cheaper of the two directions. 64g is that scaling plus
#          generous headroom; it is NOT a measured figure.
#          LI-Large (index 7): add --time=16:00:00 --mem=200g, the documented
#          envelope. The nolouvain arms: see slurm/README.md before submitting.
# TIME     06:00:00. HI-Small stage 1 is ~5 min per direction at resolution 10;
#          six hours is ~70x that, which covers the res1 arm (a coarser
#          partition leaves larger ego graphs) without asking for a slot that
#          schedules late.

#SBATCH --job-name=gargaml_meas_ibm_undir
#SBATCH --output=gargaml_meas_ibm_undir_%A_%a.out
#SBATCH --error=gargaml_meas_ibm_undir_%A_%a.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --array=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh
resolve_dataset "${1:-}"
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] measures_ibm_undir: dataset=$GARGAML_DATASET n_cpu=${GARGAML_N_CPU:-default(4)} force=${GARGAML_FORCE:-0} scratch=$GARGAML_SCRATCH results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/gargaml_undirected.py

echo Finished
