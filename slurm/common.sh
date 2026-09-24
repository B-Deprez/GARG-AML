#!/bin/bash
# Shared setup for the GARG-AML Slurm jobs. Sourced by every .slurm file, never
# executed by one -- it defines functions and exports and runs no work itself.
#
#   source slurm/common.sh
#   resolve_dataset "${1:-}"
#
# Executing it directly is a sanity check that needs no cluster:
#
#   bash slurm/common.sh
#
# which prints the valid dataset names, the synthetic array ranges, and the
# index -> name mapping, so an --array spec can be checked before submission.
#
# Bash 3.2 compatible on purpose (macOS ships 3.2): indexed arrays only, no
# associative arrays, no namerefs, no ${var^^}.

# --- Dataset names ----------------------------------------------------------
# Source of truth: the `datasets` list in scripts/gargaml_directed.py, its
# character-identical twin in scripts/gargaml_undirected.py, and the DATASETS
# list in scripts/gargaml_tree.py. This array must stay in the same order as
# those, because an array task selects its dataset by index: Python resolves
# SLURM_ARRAY_TASK_ID through src/utils/runtime.py::select_datasets, which
# indexes the script's own list. A reordering here that is not mirrored there
# silently runs the wrong dataset.
#
# Check the two agree at any time with:
#   bash slurm/common.sh --verify
IBM_DATASETS=(
  "HI-Small_bank012"      # 0  single-bank view: largest bank, 12,180 nodes
  "HI-Small_banktop50"    # 1  pooled view: 50 largest banks, 164,822 nodes
  "HI-Small_res1"         # 2  Louvain resolution 1
  "HI-Small_res5"         # 3  Louvain resolution 5
  "HI-Small"              # 4  main setting, Louvain resolution 10, 515,080 nodes
  "HI-Small_res20"        # 5  Louvain resolution 20
  "HI-Small_res50"        # 6  Louvain resolution 50
  "LI-Large"              # 7  2,054,390 nodes / 176M edges -- the big one
  "HI-Small_nolouvain"    # 8  control: no reduction at all
  "LI-Large_nolouvain"    # 9  expected infeasible -- see slurm/README.md
)

# --- Synthetic array ranges -------------------------------------------------
# construct_datasets() in scripts/gargaml_{directed,undirected}_synth.py yields
# 66 names in a deterministic order, with the size tiers contiguous. Submitting
# the tiers as separate ranges is what keeps the cheap 44 out of the queue
# behind the expensive 22.
SYNTH_TOTAL=66
SYNTH_RANGE_SMALL="0-21"     # 100 nodes      -- seconds each
SYNTH_RANGE_MEDIUM="22-43"   # 10,000 nodes   -- ~4 min median each
SYNTH_RANGE_LARGE="44-65"    # 100,000 nodes  -- ~9.8 h median each

# --- Common exports ---------------------------------------------------------
# Scratch for transient state. $VSC_SCRATCH is the un-quota'd volume; falling
# back to results/ keeps the same code path working off-cluster.
if [ -n "${VSC_SCRATCH:-}" ]; then
  export GARGAML_SCRATCH="${VSC_SCRATCH}/gargaml"
else
  export GARGAML_SCRATCH="results"
fi

# Keep pandas/numpy from oversubscribing the allocation. The measure scripts
# parallelise with multiprocessing, not BLAS threads, so extra BLAS threads only
# contend. Left unset by default -- see slurm/README.md.
# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- resolve_dataset --------------------------------------------------------
# Decide which single dataset this job runs and export it for the Python layer.
#
#   resolve_dataset "${1:-}"
#
# Order: positional arg, then $GARGAML_DATASET, then $SLURM_ARRAY_TASK_ID as an
# index into IBM_DATASETS. A name given explicitly is accepted even if it is not
# in IBM_DATASETS -- a dataset string is a tag in this repo ("HI-Small_res20",
# "HI-Small_bank012"), so the list is the menu, not the grammar. An index that
# is out of range is a hard exit 1: an array task that runs nothing and exits 0
# is the failure that is hardest to spot afterwards.
resolve_dataset() {
  local requested="${1:-}"

  if [ -n "$requested" ]; then
    export GARGAML_DATASET="$requested"
  elif [ -n "${GARGAML_DATASET:-}" ]; then
    : # already set in the environment
  elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] 2>/dev/null || \
       [ "$SLURM_ARRAY_TASK_ID" -ge "${#IBM_DATASETS[@]}" ] 2>/dev/null; then
      echo "ERROR: array index $SLURM_ARRAY_TASK_ID out of range;" \
           "valid range is 0-$(( ${#IBM_DATASETS[@]} - 1 ))." >&2
      exit 1
    fi
    export GARGAML_DATASET="${IBM_DATASETS[$SLURM_ARRAY_TASK_ID]}"
  else
    echo "ERROR: no dataset. Pass one as the first argument, set" \
         "GARGAML_DATASET, or submit as an array job." >&2
    echo "Valid names: ${IBM_DATASETS[*]}" >&2
    exit 1
  fi

  # Warn, do not fail, on a name outside the menu: composed tags are legitimate.
  local known=0 name
  for name in "${IBM_DATASETS[@]}"; do
    [ "$name" = "$GARGAML_DATASET" ] && known=1
  done
  if [ "$known" -eq 0 ]; then
    echo "NOTE: '$GARGAML_DATASET' is not one of the listed names." \
         "Proceeding -- dataset strings are tags (_res<r>, _bank<b>, _nolouvain)." >&2
  fi
}

# --- resolve_synth_index ----------------------------------------------------
# Validate the synthetic array index and report which tier it lands in. The
# Python layer does the index -> name lookup (it owns construct_datasets), so
# this only bounds-checks and labels.
resolve_synth_index() {
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: the synthetic jobs are array jobs; SLURM_ARRAY_TASK_ID is unset." >&2
    echo "Submit with --array=${SYNTH_RANGE_SMALL} (or MEDIUM/LARGE)." >&2
    exit 1
  fi
  if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] 2>/dev/null || \
     [ "$SLURM_ARRAY_TASK_ID" -ge "$SYNTH_TOTAL" ] 2>/dev/null; then
    echo "ERROR: synthetic index $SLURM_ARRAY_TASK_ID out of range 0-$(( SYNTH_TOTAL - 1 ))." >&2
    exit 1
  fi
  if [ "$SLURM_ARRAY_TASK_ID" -le 21 ]; then
    export GARGAML_SYNTH_TIER="100-node"
  elif [ "$SLURM_ARRAY_TASK_ID" -le 43 ]; then
    export GARGAML_SYNTH_TIER="10,000-node"
  else
    export GARGAML_SYNTH_TIER="100,000-node"
  fi
}

# --- Self-test --------------------------------------------------------------
# Runs only when the file is executed, never when it is sourced.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "GARG-AML Slurm harness -- slurm/common.sh"
  echo
  echo "IBM datasets (${#IBM_DATASETS[@]} entries; --array=0-$(( ${#IBM_DATASETS[@]} - 1 ))):"
  i=0
  while [ "$i" -lt "${#IBM_DATASETS[@]}" ]; do
    printf "  %2d  %s\n" "$i" "${IBM_DATASETS[$i]}"
    i=$(( i + 1 ))
  done
  echo
  echo "Synthetic array ranges (${SYNTH_TOTAL} datasets total):"
  echo "  --array=${SYNTH_RANGE_SMALL}   100-node tier      (seconds each)"
  echo "  --array=${SYNTH_RANGE_MEDIUM}  10,000-node tier   (~4 min median)"
  echo "  --array=${SYNTH_RANGE_LARGE}  100,000-node tier  (~9.8 h median)"
  echo
  echo "GARGAML_SCRATCH would be: ${GARGAML_SCRATCH}"
  echo

  if [ "${1:-}" = "--verify" ]; then
    echo "Verifying this list against the Python source of truth..."
    python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import re
shell = []
with open("slurm/common.sh") as fh:
    inside = False
    for line in fh:
        if line.startswith("IBM_DATASETS=("):
            inside = True
            continue
        if inside:
            if line.strip() == ")":
                break
            m = re.match(r'\s*"([^"]+)"', line)
            if m:
                shell.append(m.group(1))

src = open("scripts/gargaml_directed.py").read()
block = src[src.index("datasets = ["):]
block = block[:block.index("]") + 1]
py = re.findall(r'"([^"]+)"', block)

if shell == py:
    print("  OK: %d names, same order." % len(shell))
else:
    print("  MISMATCH.")
    print("  shell : %s" % shell)
    print("  python: %s" % py)
    sys.exit(1)
PYEOF
  else
    echo "Run 'bash slurm/common.sh --verify' to check the name list against"
    echo "scripts/gargaml_directed.py (needs python and the repo root as cwd)."
  fi
fi
