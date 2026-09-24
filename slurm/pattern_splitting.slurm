#!/bin/bash
# Split-pattern diagnostic -- scripts/pattern_splitting.py.
#
# WHAT     For every laundering attempt in the patterns file, at every
#          resolution in the sweep, records whether the attempt stayed in one
#          community, what fraction of its edges survived, and whether any of
#          its source->mule->target 2-paths survived. Once the 2-path is gone,
#          GARG-AML cannot see the pattern at any threshold. Writes
#          <dataset>_pattern_splitting[_summary].csv and the pooled
#          pattern_splitting_summary.csv.
# SUBMIT   slurm/submit.sh slurm/pattern_splitting.slurm HI-Small
# ARRAY    None. The six resolutions share one graph build, so splitting them
#          across tasks would rebuild the graph six times -- keep it serial.
# PREREQ   None -- reads only data/. Independent of stage 1.
# RESUMES  No, but it is minutes for HI-Small, so a rerun is cheap.
# CPUS     4 -- serial, with headroom for the pandas passes.
# MEM      64g. community_map does G.copy().to_undirected(), holding roughly
#          three graph objects transiently on a 5M-edge graph; headroom over
#          that, not a measured peak.
# TIME     04:00:00. HI-Small takes a few minutes, most of it before the first
#          resolution. LI-Large is the multi-hour arm (176M edges) -- give it
#          --time=16:00:00 --mem=200g.
#
# CAUTION  The pooled pattern_splitting_summary.csv concatenates over the whole
#          DATASETS list. Running one dataset per job means each run overwrites
#          that file with its own single-dataset pool. Run the datasets in one
#          job, or treat the per-dataset files as the real output and rebuild
#          the pool afterwards.

#SBATCH --job-name=gargaml_pattern_splitting
#SBATCH --output=gargaml_pattern_splitting.out
#SBATCH --error=gargaml_pattern_splitting.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh
resolve_dataset "${1:-}"
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] pattern_splitting: dataset=$GARGAML_DATASET resolutions=default(none,1,5,10,20,50) results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/pattern_splitting.py

echo Finished
