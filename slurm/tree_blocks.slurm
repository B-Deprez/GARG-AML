#!/bin/bash
# Stage 2, block-only ablation -- scripts/gargaml_tree_blocks.py.
#
# WHAT     A thin entry point onto gargaml_tree.py's shared path with
#          CONFIG = "blocks": the block densities and sizes on their own,
#          without the GARG-AML score.
# SUBMIT   slurm/submit.sh slurm/tree_blocks.slurm HI-Small
#          sbatch slurm/tree_blocks.slurm [<dataset>]   (HI-Small when omitted)
# ARRAY    None.
# PREREQ   The same stage-1 measures as slurm/tree.slurm, same dataset name.
# RESUMES  Per section, as in tree.slurm: each of its two sections (blocks,
#          undirected then directed) writes its files when it finishes, a kill
#          loses only the one in progress, and a resubmission refits both.
# CPUS/MEM 18 / 64g. The same spawn pool as tree.slurm (it goes through the same
#          run_config): width from SLURM_CPUS_PER_TASK or GARGAML_N_CPU, outputs
#          byte-identical for any width. 18 rather than 36 because 6 + 18
#          features over two sections is about a quarter of tree.slurm's
#          cell-feature cost, and a quarter node is easier to schedule. No
#          graph or Louvain is built, and each worker's matrix is narrower than
#          tree.slurm's, so an estimated 0.5-1 GB per worker at most, 9-18 GB
#          for 18. Headroom over that, not a measured peak.
# TIME     02:00:00. By tree.slurm's cost fit (~16.5 s per cell-feature, 46
#          fitted cells per section) this is an estimated ~5 h of single-core
#          fitting -- more than 02:00:00 allowed when the fits were serial -- so
#          roughly 20-40 min on 18 workers, plus two label builds from the
#          transaction CSV (unmeasured). Replace with the per-section "fitted
#          <n> splits" log lines from the first run.
#
# NOTE     The dataset is the optional first argument (or GARGAML_DATASET),
#          HI-Small by default. The job reads the fold partition tree.slurm
#          wrote for that dataset and fails if its fold count differs from
#          GARGAML_N_FOLDS, so the blocks results stay paired with the other
#          configs; with no partition on disk it writes one.

#SBATCH --job-name=gargaml_tree_blocks
#SBATCH --output=slurm/logs/gargaml_tree_blocks.out
#SBATCH --error=slurm/logs/gargaml_tree_blocks.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

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
resolve_dataset "${1:-${GARGAML_DATASET:-HI-Small}}"
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] tree_blocks: dataset=$GARGAML_DATASET config=blocks n_folds=${GARGAML_N_FOLDS:-default(5)} results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/gargaml_tree_blocks.py

echo Finished
