#!/bin/bash
# Stage 2, block-only ablation (task 3) -- scripts/gargaml_tree_blocks.py.
#
# WHAT     A thin entry point onto gargaml_tree.py's shared path with
#          CONFIG = "blocks", i.e. the (b) feature group (block densities and
#          sizes) that the published model never received.
# SUBMIT   sbatch slurm/tree_blocks.slurm
# ARRAY    None.
# PREREQ   The same stage-1 measures as slurm/tree.slurm, same dataset name.
# RESUMES  NO -- same end-of-run metric writes as tree.slurm.
# CPUS/MEM 8 / 64g, identical reasoning to tree.slurm: one feature config
#          instead of four, so this is strictly the cheaper job.
# TIME     02:00:00 -- one config of tree.slurm's four.
#
# NOTE     gargaml_tree_blocks.py holds its dataset choice inside main(), not
#          at module scope, so there is no env var to select a dataset here.
#          Edit the script's own `dataset = "HI-Small"` constant to run a
#          different one.

#SBATCH --job-name=gargaml_tree_blocks
#SBATCH --output=gargaml_tree_blocks.out
#SBATCH --error=gargaml_tree_blocks.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] tree_blocks: running scripts/gargaml_tree_blocks.py's own dataset constant, config=blocks results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/gargaml_tree_blocks.py

echo Finished
