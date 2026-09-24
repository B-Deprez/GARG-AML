#!/bin/bash
# Reporting -- scripts/build_tables.py. Runs last.
#
# WHAT     Reads every tidy *_metrics.csv on disk and renders the result tables
#          as LaTeX + CSV, plus table_coverage.csv. It recomputes nothing, which
#          is why it takes seconds and can be rerun after every job finishes. It
#          prints a coverage report first, skips tables with no rows, and renders
#          unfittable cells as "--"; that gap reporting is intended, so do not
#          suppress it.
# SUBMIT   Chain it with afterany, not afterok, so one failed arm does not block
#          the tables:
#            slurm/submit.sh slurm/collect.slurm all --dependency=afterany:$j1:$j2
# ARRAY    None.
# PREREQ   Whatever model jobs you want reflected. It tolerates missing arms by
#          design.
# RESUMES  Not applicable -- idempotent, rerun it freely.
# CPUS     1 -- pandas only, no graph work.
# MEM      8g -- it reads small tidy CSVs, not transaction files.
# TIME     00:15:00 -- the tables take seconds; this is slack for the timing
#          concatenation below.
#
# NOTE     build_tables.py does not read the timing files, so GARG-AML runtime
#          never enters its tables. The step at the end of this job gathers the
#          per-task timing rows into <dir>/timing_all.csv for runtime figures,
#          where <dir> is $GARGAML_RESULTS_DIR. src/utils/reporting.py also
#          parses a direction-less tidy filename, falling back to its own
#          `direction` column, so the partial-observability metrics reach a
#          table too; see slurm/README.md.

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
export GARGAML_RESULTS_DIR="${GARGAML_RESULTS_DIR:-results-revision}"

echo "[$(date)] collect: building tables from every *_metrics.csv in $GARGAML_RESULTS_DIR/ results_dir=$GARGAML_RESULTS_DIR"

python -u scripts/build_tables.py

# Concatenate the per-task timing files into one frame. Each task writes its own
# file under <dir>/timing/ carrying dataset, direction, seconds, worker count,
# job id, host and timestamp. The <dir>/time_results_*.txt files are written as
# well -- append-shared by three scripts, one "<dataset>: <secs>" line each --
# and are what VisualisationRunTime.ipynb reads.
# The heredoc delimiter is quoted ('PYEOF'), so shell variables do not expand
# inside it -- GARGAML_RESULTS_DIR is read from the environment below instead of
# being interpolated into the script text.
python -u - <<'PYEOF'
import glob
import os
import pandas as pd

results_dir = os.environ.get("GARGAML_RESULTS_DIR", "results")
files = sorted(glob.glob(os.path.join(results_dir, "timing", "*.csv")))
if not files:
    print("[timing] no "+results_dir+"/timing/*.csv yet -- nothing to collect.")
else:
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame.sort_values(["dataset", "direction", "timestamp"])
    out = os.path.join(results_dir, "timing_all.csv")
    frame.to_csv(out, index=False)
    print("[timing] %d task files -> %s (%d rows)" % (len(files), out, len(frame)))
PYEOF

echo Finished
