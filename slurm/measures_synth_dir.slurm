#!/bin/bash
# Stage 1, synthetic directed block measures -- scripts/gargaml_directed_synth.py.
#
# WHAT     One synthetic dataset per array task: builds the graph from
#          data/edge_data_<name>.csv, Louvain-reduces it at the default
#          resolution 10, computes the nine directed block measures per node.
#          Writes results/<dataset>_GARGAML_directed.csv.
# SUBMIT   Submit the size tiers SEPARATELY -- their costs differ by four orders
#          of magnitude and a single --array=0-65 queues the cheap 44 behind the
#          expensive 22:
#            sbatch --array=0-21  --time=00:30:00 --mem=8g   slurm/measures_synth_dir.slurm
#            sbatch --array=22-43 --time=02:00:00 --mem=16g  slurm/measures_synth_dir.slurm
#            sbatch --array=44-65 --time=24:00:00 --mem=64g  slurm/measures_synth_dir.slurm
# ARRAY    --array=<i> indexes construct_datasets(), 66 entries. VERIFIED to be a
#          bijection over 0-65 by dry-running the decoder, with CONTIGUOUS size
#          tiers: 0-21 = 100 nodes, 22-43 = 10,000, 44-65 = 100,000.
#          This array is the single biggest win in the harness: a ~9-day serial
#          loop becomes roughly the wall time of its slowest dataset.
# PREREQ   data/edge_data_<name>.csv for each name (all 66 present locally).
# RESUMES  YES, at dataset granularity -- see measures_ibm_dir.slurm. Because one
#          task is one dataset here, a killed task costs only that dataset.
# CPUS     8 -> the script's 4-worker pool (min(4, cpu_count()//2)). See
#          measures_ibm_dir.slurm for why this is not auto-detected.
# MEM      Defaults to 16g, which suits the 10,000-node tier. Override per tier
#          as above: 8g is ample at 100 nodes; the 100,000-node tier densifies
#          larger ego graphs and gets 64g.
# TIME     Defaults to 06:00:00, correct for the 10,000-node tier (~4 min
#          median). The 100,000-node tier has a ~9.8 h MEDIAN, so 6 h would kill
#          over half of it -- give that range --time=24:00:00 explicitly. The
#          median is what the archived timing logs record; the MAX is what sizes
#          a wall clock, and the archives cannot be attributed to a run (three
#          scripts append to the same file), so 24 h is deliberate headroom
#          rather than a measured bound.

#SBATCH --job-name=gargaml_meas_synth_dir
#SBATCH --output=gargaml_meas_synth_dir_%A_%a.out
#SBATCH --error=gargaml_meas_synth_dir_%A_%a.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --array=0-21
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=16g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh
resolve_synth_index
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] measures_synth_dir: index=$SLURM_ARRAY_TASK_ID tier=$GARGAML_SYNTH_TIER n_cpu=${GARGAML_N_CPU:-default(4)} force=${GARGAML_FORCE:-0} results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/gargaml_directed_synth.py

echo Finished
