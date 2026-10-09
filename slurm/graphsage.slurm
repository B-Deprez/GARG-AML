#!/bin/bash
# GraphSAGE baseline -- scripts/graphsage_baseline.py. The only GPU job.
#
# WHAT     Trains a 2-layer GraphSAGE per (feature config, cut-off, target,
#          fold), early-stopping on validation AUC-PR, and records fit and
#          inference timings plus peak host and GPU memory. Both feature configs
#          are built from the raw transaction file, never from the GARG-AML
#          measure CSVs: feeding it GARG-AML scores, block densities or block
#          sizes would destroy the ablation.
# SUBMIT   slurm/submit.sh slurm/graphsage.slurm HI-Small
# ARRAY    None. One dataset per invocation by design. The full sweep runs as
#          one job per target (GARGAML_TARGETS, each with its own
#          GARGAML_OUTPUT_DIR), merged afterwards with GARGAML_MERGE_FROM --
#          see slurm/README.md. GARGAML_CONFIGS shards the two feature configs
#          the same way.
# PREREQ   The tree job for the same dataset name -- it writes
#          <dataset>_folds.csv, which this job reads for its fold count and fold
#          membership. It has no fold count of its own, so whatever the tree ran
#          with is what this inherits; a 2-fold partition silently gives a
#          2-fold GraphSAGE run.
#            t=$(slurm/submit.sh slurm/tree.slurm HI-Small --array=4)
#            slurm/submit.sh slurm/graphsage.slurm HI-Small --dependency=afterok:$t
#          For LI-Large also slurm/graphsage_prep.slurm, whose caches keep the
#          label build off this node (see MEM).
# RESUMES  Partially. train_fold checkpoints every epoch with a signature check
#          and resumes from it, so a killed fit restarts mid-fold rather than
#          from scratch. But the tidy/runs/summary CSVs are written only after
#          the last fold, so a wall-time kill loses the metrics of every
#          completed fit while keeping their checkpoints. Size --time to finish
#          the whole sweep; do not plan on a restart to collect the metrics.
# GPU      One GPU, and --nodes=1 explicitly: wice's GPU partitions reject a
#          submission that gives no node count, unlike the batch partition
#          every other job here uses.
#          A 2-layer GraphSAGE with a small hidden dimension is one
#          model on one device; the script has no data-parallel path, so a
#          second GPU would sit idle. GARGAML_REQUIRE_GPU=1 is set below and is
#          the point of this job: get_device() otherwise falls back to CPU
#          silently, and it prefers MPS before CUDA, so the guard asserts CUDA
#          specifically. Without it an allocated A100 can burn the full wall
#          clock at CPU speed and report timings that are not what the paper
#          claims.
# CPUS     8. HI-Small uses num_workers=0 (full-graph inference, infer_batch_size
#          None); LI-Large uses num_workers=4 plus the main process, and on wice
#          the cpu request follows the memory share, so 8 covers both without
#          over-requesting.
# MEM      64g for HI-Small. Arithmetic: edge_index is 2 x 5,078,345 x 8 bytes =
#          ~81 MB, node features 515,080 x ~20 float32 = ~41 MB, and the graph is
#          built from a 475 MB CSV read whole by pandas. The host-side peak is
#          dominated by that build, not by the tensors. 64g is headroom.
#          LI-Large: run slurm/graphsage_prep.slurm first, then
#          --time=16:00:00 --mem=48g. 200g cannot be had on this partition:
#          gpu_a100 allows at most 126,000 MiB and 18 cores per GPU, and sbatch
#          rejects more. What needed it was the label build -- 166.6 GiB MaxRSS
#          (label_distribution.py's identical call, job 62206918) -- which the
#          prep job does on a CPU node. With its caches this job loads
#          edge_index (at most ~5.6 GB int64, copied once into a tensor), the
#          features (2,054,390 x ~20 x 4 = ~164 MB) and the folds file. The
#          measured host peak of a cached run was 12.4 GB (peak_host_mb in
#          LI-Large_graphsage_runs.csv), so 48g is ~4x that; 120g, the first
#          choice, was an unmeasured guess. On wice the cpu count follows the
#          memory share (120g gave 18 cores, 48g gives ~7), and the job needs 5
#          (4 loader workers + the main process), so do not go below ~40g.
#          Check sacct's MaxRSS after the first job and raise it for a target
#          that dies with an out-of-memory error. Without the caches it
#          rebuilds the labels itself and runs out of memory at any size. Its
#          num_workers=4 assumes pyg-lib is present (see the script's
#          docstring); without it the neighbour sampler is slower but not heavier.
# TIME     16:00:00 for one target. The full sweep -- the tree models' grid,
#          540 fits on HI-Small and 360 on LI-Large -- does not fit one job,
#          which is why it runs one target per job. One target is 60 fits on
#          HI-Small and 40 on LI-Large; at LI-Large's measured 17-23 s/epoch
#          the worst case, every fit at the 50-epoch cap, is ~11 h. Early
#          stopping and zero-positive skips make most jobs far shorter.

#SBATCH --job-name=gargaml_graphsage
#SBATCH --output=slurm/logs/gargaml_graphsage_%j.out
#SBATCH --error=slurm/logs/gargaml_graphsage_%j.err

#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
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

# Fail immediately rather than training on CPU with a GPU allocated.
export GARGAML_REQUIRE_GPU=1

echo "[$(date)] graphsage: dataset=$GARGAML_DATASET configs=${GARGAML_CONFIGS:-default(topology,attributes)} targets=${GARGAML_TARGETS:-default} output_dir=${GARGAML_OUTPUT_DIR:-$GARGAML_RESULTS_DIR} require_gpu=$GARGAML_REQUIRE_GPU checkpoints=${GARGAML_CHECKPOINT_DIR:-\$VSC_SCRATCH/gargaml/checkpoints} results_dir=$GARGAML_RESULTS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u scripts/graphsage_baseline.py

echo Finished
