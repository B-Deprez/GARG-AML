#!/bin/bash
# Experiment status -- scripts/status.py.
#
# WHAT     Reads $GARGAML_RESULTS_DIR and reports, per experiment, whether its
#          output is on disk and complete, which outputs are stale (single-split
#          runs, pre-c5fba86 directed measures, a GraphSAGE attributes run that
#          still carries the timing features), which jobs are in the queue, and
#          what can be submitted next. Writes nothing but <dir>/status.csv, one
#          row per experiment. It recomputes nothing, so rerun it freely.
# SUBMIT   slurm/submit.sh slurm/status.slurm all
#          then read slurm/logs/gargaml_status.out (overwritten on every run).
#          Extra arguments go through the environment, not the command line:
#            STATUS_ARGS="--long" slurm/submit.sh slurm/status.slurm all
#            STATUS_ARGS="--dir results results-revision" slurm/submit.sh slurm/status.slurm all
#          The positional dataset only satisfies submit.sh; this job ignores it.
#          On a login node the same report is one line, no job needed:
#            source activate gargaml && python scripts/status.py --dir results-revision
# ARRAY    None.
# PREREQ   None. It tolerates any mix of finished and missing arms by design.
#          Chain it last with afterany to see where a wave ended up.
# RESUMES  Not applicable -- idempotent.
# CPUS     1 -- pandas only.
# MEM      8g. The widest read is LI-Large's directed measures (two columns,
#          ~2M rows) for the pre-fix check; the tidy metrics are small.
# TIME     00:30:00. About a minute: 66 synthetic measure files plus the IBM
#          ones are opened to check their values. STATUS_ARGS="--no-values"
#          skips those reads.
#
# NOTE     The queue section calls `squeue -M wice`. A job submitted through
#          slurm/submit.sh is named <script>_<dataset> and is matched to its
#          experiment by name; a bare `sbatch` keeps the script's default name
#          and is only matched when it is an array job, by task id. Jobs it
#          cannot place are named on the "Queue:" line, never dropped.

#SBATCH --job-name=gargaml_status
#SBATCH --output=slurm/logs/gargaml_status.out
#SBATCH --error=slurm/logs/gargaml_status.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML || exit 1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml
# Fail fast from here on, so a crash ends the job as FAILED (FAIL mail) rather
# than COMPLETED via the closing echo. Set after activation, which is not -e
# safe; the activation is checked explicitly instead.
[ "${CONDA_DEFAULT_ENV:-}" = "gargaml" ] || { echo "ERROR: conda env gargaml did not activate" >&2; exit 1; }
set -eo pipefail

export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] status: results_dir=$GARGAML_RESULTS_DIR args=${STATUS_ARGS:-none}"

# STATUS_ARGS is deliberately unquoted: it is a list of flags, and the shell has
# to split it. None of its values contains a space.
python -u scripts/status.py ${STATUS_ARGS:-} --csv "$GARGAML_RESULTS_DIR/status.csv"

echo Finished
