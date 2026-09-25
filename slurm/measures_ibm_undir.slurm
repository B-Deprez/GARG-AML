#!/bin/bash
# Stage 1, IBM undirected block measures -- scripts/gargaml_undirected.py.
#
# WHAT     Builds the transaction graph, applies the Louvain reduction the
#          dataset name asks for, and computes the three undirected block
#          measures per node. Writes
#          $GARGAML_RESULTS_DIR/<dataset>_GARGAML_undirected.csv.
# SUBMIT   sbatch --array=4 slurm/measures_ibm_undir.slurm
#          slurm/submit.sh slurm/measures_ibm_undir.slurm HI-Small --array=4
# ARRAY    --array=<i> indexes slurm/common.sh's IBM_DATASETS; index 4 is the
#          main HI-Small setting. Submit only the arms you want:
#            --array=0,1     single-bank views     (cheap)
#            --array=2,3,5,6 Louvain sweep         (~5 min each)
#            --array=4       HI-Small              (~5 min)
#            --array=7       LI-Large              needs the overrides below
#            --array=8       HI-Small_nolouvain    needs the overrides below
#            --array=9       LI-Large_nolouvain    expected infeasible, see README
#            --array=10,11,12 hub removal instead of Louvain; same cost as index 8
# PREREQ   None -- stage 1 reads only data/. This is the head of the chain.
# RESUMES  Yes, at dataset granularity. The script skips a dataset whose output
#          CSV already exists, so resubmitting the same array is the recovery
#          procedure. Writes are atomic (temp + os.replace), so a wall-time kill
#          cannot leave a truncated file that the skip mistakes for finished.
#          GARGAML_FORCE=1 recomputes regardless.
# CPUS     8, sized to the code rather than to the node. The script caps its
#          pool at min(4, cpu_count()//2), so 8 cores are what yield its 4
#          workers; 16 would leave 12 idle, and a single cpu gives
#          Pool(processes=0), which raises. The worker count is fixed on
#          purpose: it sets the runtimes the scalability figure reports.
# MEM      64g for the HI-Small arms. construct_IBM_graph reads the whole 475 MB
#          CSV with pd.read_csv (no chunksize); this path then holds only G and
#          G_reduced, so it is the cheaper of the two directions. Headroom over
#          HI-Small (515,080 nodes / 5,078,345 edges), not a measured peak.
#          LI-Large (index 7): add --time=16:00:00 --mem=200g.
#          The nolouvain arms: see slurm/README.md before submitting.
# TIME     06:00:00. HI-Small stage 1 is ~5 min per direction at resolution 10;
#          six hours also covers the res1 arm, where a coarser partition leaves
#          larger ego graphs.

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
