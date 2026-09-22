#!/bin/bash
# Reporting -- scripts/build_tables.py. Runs LAST.
#
# WHAT     Reads every tidy *_metrics.csv on disk and renders the revision's
#          tables as LaTeX + CSV, plus results/table_coverage.csv. Recomputes
#          nothing, which is why it takes seconds and can be rerun after every
#          job finishes. It prints a coverage report FIRST, skips tables with no
#          rows, and renders unfittable cells as "--" -- that gap-reporting is
#          the behaviour the revision wants, so do not suppress it.
# SUBMIT   Chain it with afterany, NOT afterok, so one failed arm does not block
#          the tables:
#            slurm/submit.sh slurm/collect.slurm all --dependency=afterany:$j1:$j2
# ARRAY    None.
# PREREQ   Whatever model jobs you want reflected. It tolerates missing arms by
#          design.
# RESUMES  Not applicable -- idempotent, rerun it freely.
# CPUS     1 -- pandas only, no graph work.
# MEM      8g -- it reads small tidy CSVs, not transaction files.
# TIME     00:15:00 -- the docstring says seconds; this is slack for the timing
#          concatenation below.
#
# NOTE     build_tables.py does NOT read the timing files, so GARG-AML runtime
#          never enters its tables. The step at the end of this job gathers the
#          per-task timing rows into results/timing_all.csv for Fig. 6-style
#          work. Two metrics files are silently dropped by its filename regex
#          (the partial_observability pair carries no direction token), so
#          task 5's appendix numbers do not reach a table -- see slurm/README.md.

#SBATCH --job-name=gargaml_collect
#SBATCH --output=gargaml_collect.out
#SBATCH --error=gargaml_collect.err

#SBATCH --cluster=wice
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=8g
#SBATCH --account=lp_verbekelab
#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user bruno.deprez@kuleuven.be

cd $VSC_DATA/GARGAML/11.2Code/GARG-AML

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"

source activate gargaml

source slurm/common.sh

echo "[$(date)] collect: building tables from every *_metrics.csv in results/"

python -u scripts/build_tables.py

# Concatenate the per-task timing files into one frame. Each task writes its own
# file under results/timing/ carrying dataset, direction, seconds, worker count,
# job id, host and timestamp -- unlike the legacy results/time_results_*.txt,
# which is append-shared by three scripts and records only "<dataset>: <secs>".
# The legacy files are still written, so VisualisationRunTime.ipynb is unaffected.
python -u - <<'PYEOF'
import glob
import pandas as pd

files = sorted(glob.glob("results/timing/*.csv"))
if not files:
    print("[timing] no results/timing/*.csv yet -- nothing to collect.")
else:
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame.sort_values(["dataset", "direction", "timestamp"])
    out = "results/timing_all.csv"
    frame.to_csv(out, index=False)
    print("[timing] %d task files -> %s (%d rows)" % (len(files), out, len(frame)))
PYEOF

echo Finished
