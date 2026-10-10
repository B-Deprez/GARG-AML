"""
Which experiments have run, which have not, and which cannot run yet.

Reads a results directory and reports, per experiment, whether its output is
on disk and whether it is complete -- then says what to submit next. Nothing is
recomputed and nothing is written (except the optional ``--csv``), so it takes
seconds and can be rerun after every job finishes.

Run from the repository root::

    python scripts/status.py                       # $GARGAML_RESULTS_DIR, else results/
    python scripts/status.py --dir results-revision
    python scripts/status.py --dir results results-revision   # one report each
    python scripts/status.py --long                # every experiment, with notes
    python scripts/status.py --no-values           # skip the value-level checks (faster)
    python scripts/status.py --csv status.csv      # one row per experiment

On the cluster: ``slurm/submit.sh slurm/status.slurm all``, then read
``slurm/logs/gargaml_status.out``.

States
------
``done``   every expected output is on disk and complete.
``part``   some outputs are on disk; the detail says what is missing.
``STALE``  outputs exist but are known to be superseded: a single-split run
           from before the cross-validation, a result file with no 0.0 cut-off
           row, directed measures computed before commit ``c5fba86``, a
           GraphSAGE ``attributes`` run that still carries the timing features,
           or a result built on top of any of those. Rerun to replace.
``ready``  nothing on disk yet, and every input it needs is there: submit it.
``wait``   nothing on disk yet, and an input it needs is missing: the detail
           says which.
``RUN`` / ``PEND``  a job of this experiment is in the queue (``squeue``).
``-``      not part of the plan for that dataset.

Ground truth is what got written, not the exit code -- a task can exit 0 having
written nothing, and ``sacct`` ages out -- so a job that is no longer queued and
left no file shows up as ``ready`` again, not as done.

What it checks, and what it does not
------------------------------------
Completeness is judged against the plan the scripts themselves declare: the IBM
dataset list, targets and cut-offs from ``gargaml_tree.py``, the 66-dataset
synthetic grid from ``construct_datasets``, GraphSAGE's per-dataset sweep from
``graphsage_baseline.py`` (``DATASET_SETTINGS`` is deliberately *not* used for
the tree files: an unfittable cell is written as a NaN row with a status, so a
complete file has every cell). Those are read out of the script source rather
than imported, so this script needs no model dependencies and cannot drift from
them. Cells that are ``skipped`` for a stated reason (outside the dataset's
sweep, no positives, too few positives) are by design, not missing; ``--long``
counts them.

It does not judge the numbers. A ``done`` GraphSAGE row means the cells are
there, not that the fit was good.
"""

import argparse
import ast
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

os.chdir("./")
sys.path.append("./")

from src.utils.evaluation import CUT_OFFS, HEADLINE_CUTOFFS
from src.utils.features import FEATURE_CONFIGS, config_suffix, is_direction_free
from src.utils.graph_processing import parse_hubs, strip_resolution
from src.utils.runtime import resolve_results_dir

DIRECTIONS = ["undirected", "directed"]

# Datasets that carry the whole experiment set (base scores, GraphSAGE, label
# tables, pattern splitting, FlowScope). The Louvain sweep, hub removal and the
# bank views get measures and trees only.
CORE_DATASETS = ["HI-Small", "LI-Large"]

# gargaml_IF.py hardcodes its dataset in main().
IF_DATASET = "HI-Small"

# Expected to be infeasible (slurm/README.md, Known gaps): reported, but never
# counted against the run.
OPTIONAL_DATASETS = {"LI-Large_nolouvain"}

# The one `slurm/distribution_scores.slurm` argument that runs every score-only
# arm in a single job (scripts/distribution_scores.py, LOUVAIN_SWEEP).
SWEEP_JOB = "louvain_sweep"

# Stage 1 cost per synthetic size tier, from slurm/README.md's submission order.
SYNTH_COST = {100: ("00:30:00", "8g"), 10000: ("02:00:00", "16g"),
              100000: ("24:00:00", "64g")}

# The fold partition repeats one block of rows per (cut-off, target); the first
# block already carries every fold, so there is no need to read the rest of a
# multi-gigabyte file to count them.
FOLDS_SAMPLE_ROWS = 3_000_000

# Value-level check for the directed measures: an empty block 12 cannot score a
# full 1.0 in the fixed code (commit c5fba86, see CLAUDE.md).
PRE_FIX = "size_12 == 0 and measure_12 == 1"

STATE_WORD = {"DONE": "done", "PARTIAL": "part", "STALE": "STALE", "MISSING": "miss", "READY": "ready",
              "BLOCKED": "wait", "RUNNING": "RUN", "QUEUED": "PEND", "NA": "-"}
CELL = {"DONE": "done", "STALE": "STALE", "PARTIAL": "part", "MISSING": "miss"}
MATRIX = ["meas-D", "meas-U", "folds", "tree-D", "tree-U", "base-D", "base-U", "sage", "IF"]


# ---------------------------------------------------------------------------
# The plan, read out of the scripts
# ---------------------------------------------------------------------------

def _source(script):
    path = os.path.join("scripts", script)
    try:
        with open(path) as fh:
            return ast.parse(fh.read(), filename=path)
    except OSError as exc:
        raise SystemExit("status.py: cannot read "+path+" ("+str(exc)+"). Run from the repository root.")


def script_constant(script, name, namespace=None):
    """Module-level ``name = <expression>`` of ``scripts/<script>``, without importing it.

    The first assignment that evaluates wins, because several scripts rebind
    the constant afterwards (``DATASETS = select_datasets(DATASETS)``), which
    does not evaluate here and is exactly the override this must ignore.
    """
    for node in _source(script).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return eval(compile(ast.Expression(node.value), script, "eval"),
                            dict(namespace or {}))
            except Exception:  # noqa: BLE001 - a later assignment may be the literal one
                continue
    raise SystemExit("status.py: cannot read "+name+" from scripts/"+script+
                     ". The plan is taken from there; update status.py if it moved.")


def script_function(script, name):
    """The module-level function ``name`` of ``scripts/<script>``, compiled alone."""
    for node in _source(script).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {}
            exec(compile(ast.Module([node], []), script, "exec"), namespace)  # noqa: S102
            return namespace[name]
    raise SystemExit("status.py: cannot find "+name+"() in scripts/"+script+".")


@dataclass
class Plan:
    ibm: list
    targets: list
    synth: list
    graphsage: dict
    graphsage_suffix: dict
    institutions: list
    view_base: str


def load_plan():
    targets = script_constant("gargaml_tree.py", "TARGET_COLUMNS")
    names = {"CUT_OFFS": CUT_OFFS, "HEADLINE_CUTOFFS": HEADLINE_CUTOFFS,
             "TARGET_COLUMNS": script_constant("graphsage_baseline.py", "TARGET_COLUMNS")}
    return Plan(
        ibm=script_constant("gargaml_tree.py", "DATASETS"),
        targets=targets,
        synth=script_function("gargaml_directed_synth.py", "construct_datasets")(),
        graphsage=script_constant("graphsage_baseline.py", "DATASETS", names),
        graphsage_suffix=script_constant("graphsage_baseline.py", "CONFIG_SUFFIXES"),
        institutions=script_constant("partial_observability.py", "INSTITUTIONS"),
        view_base=script_constant("partial_observability.py", "DATASET"),
    )


# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """One experiment: its outputs on disk, what it needs, how to submit it.

    ``status`` describes the outputs alone (DONE, PARTIAL, STALE, MISSING, NA);
    ``state`` is what the report shows, which adds whether a MISSING one can run
    yet (READY / BLOCKED) and whether it is already queued (RUNNING / QUEUED).
    """
    key: str
    label: str
    group: str                       # "ibm" | "synth" | "extra"
    status: str
    detail: str = ""
    note: str = ""                   # shown with --long only
    needs: tuple = ()                # keys of units whose outputs this reads
    taints: dict = field(default_factory=dict)   # column -> keys; DONE becomes STALE if one is
    cmd: str = ""                    # may hold several lines
    optional: bool = False
    quiet: bool = False              # a dependency marker: no next step of its own
    cols: dict = field(default_factory=dict)     # matrix column -> text; "miss" is filled in by resolve()
    family: str = ""                 # slurm script basename, for the queue overlay
    dataset: str = None
    indices: frozenset = frozenset()
    groups: tuple = ()               # extra job names (the dataset part) that run this unit
    state: str = ""
    waiting_on: tuple = ()
    jobs: tuple = ()


def bucket(status):
    """Group a ``skipped: ...`` status by cause, so the counts stay readable."""
    if "not in this dataset's sweep" in status:
        return "outside sweep"
    if "no positive" in status or "1 class" in status or "one class" in status:
        return "one class"
    if "least populated class" in status:
        return "too few positives"
    if "incomplete out-of-fold" in status:
        return "incomplete OOF"
    return status[:40]


@dataclass
class Tidy:
    cells: set
    cutoffs: set
    folds: set
    pooled: bool
    ok_cells: int
    skipped: Counter
    mtime: float


def read_tidy(path):
    """What a tidy ``*_metrics.csv`` holds, at the granularity completeness needs."""
    wanted = {"cutoff", "target", "fold", "model", "status"}
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    missing = wanted - {"fold"} - set(df.columns)
    if missing:
        raise ValueError("no "+", ".join(sorted(missing))+" column")
    if "fold" not in df.columns:              # a single-split file, from before the CV
        df["fold"] = float("nan")
    rows = df[df["model"].notna()]            # the imbalance rows carry no model
    ok = rows["status"] == "ok"
    return Tidy(
        cells=set(zip(rows["cutoff"].astype(float), rows["target"])),
        cutoffs=set(rows["cutoff"].astype(float)),
        folds={int(f) for f in rows["fold"].dropna().unique() if f >= 0},
        pooled=bool((rows["fold"] == -1).any()),
        ok_cells=rows.loc[ok, ["cutoff", "target"]].drop_duplicates().shape[0],
        skipped=Counter(bucket(s) for s in rows.loc[~ok, "status"].astype(str)),
        mtime=os.path.getmtime(path),
    )


def combine(statuses):
    """One status for a unit made of several outputs."""
    statuses = list(statuses)
    if all(s == "DONE" for s in statuses):
        return "DONE"
    if all(s == "MISSING" for s in statuses):
        return "MISSING"
    if all(s in ("DONE", "STALE") for s in statuses):
        return "STALE"
    return "PARTIAL"


def compress(indices):
    """``[44, 45, 46, 50]`` -> ``"44-46,50"``, the form ``--array`` takes."""
    indices, spans = sorted(indices), []
    for i in indices:
        if spans and i == spans[-1][1] + 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return ",".join(str(a) if a == b else str(a)+"-"+str(b) for a, b in spans)


def day(timestamp):
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


class Report:
    """Every unit for one results directory."""

    def __init__(self, results_dir, plan, check_values=True):
        self.dir = results_dir
        self.plan = plan
        self.check_values = check_values
        self.units = []
        self._folds = {}

    def path(self, name):
        return os.path.join(self.dir, name)

    def exists(self, name):
        path = self.path(name)
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def add(self, unit):
        self.units.append(unit)
        return unit

    def folds(self, dataset):
        """Fold ids in ``<dataset>_folds.csv``, or ``None`` when it is not there."""
        if dataset not in self._folds:
            ids = None
            if self.exists(dataset+"_folds.csv"):
                sample = pd.read_csv(self.path(dataset+"_folds.csv"), usecols=["fold"],
                                     nrows=FOLDS_SAMPLE_ROWS)
                ids = sorted(int(f) for f in sample["fold"].dropna().unique())
            self._folds[dataset] = ids
        return self._folds[dataset]

    def section(self, name, targets, cut_offs=None, cv=True, dataset=None):
        """``(status, problems, tidy)`` for one tidy metrics file.

        Every (cut-off, target) cell must be there; a cell skipped for a stated
        reason still has its rows, so it counts as present. ``cv=False`` is for
        a model that is a single split by design (the isolation forest): there
        the absence of folds is not a defect.
        """
        if not self.exists(name):
            return "MISSING", ["no file"], None
        try:
            tidy = read_tidy(self.path(name))
        except Exception as exc:  # noqa: BLE001 - report, do not crash the report
            return "PARTIAL", ["unreadable: "+str(exc)], None

        sweep = CUT_OFFS if cut_offs is None else cut_offs
        expected = {(float(c), t) for c in sweep for t in targets}
        problems, stale = [], []
        # A file from before 2026-09-24 has no 0.0 rows at all: one stale file,
        # not a pile of missing cells.
        if 0.0 in sweep and 0.0 not in tidy.cutoffs:
            stale.append("no 0.0 cut-off rows (written before 2026-09-24)")
            expected = {cell for cell in expected if cell[0] != 0.0}
        if expected - tidy.cells:
            problems.append(str(len(tidy.cells & expected))+"/"+str(len(expected))+" cells")
        if cv:
            if not tidy.folds:
                stale.append("single split, no fold column (predates the CV)")
            else:
                want = self.folds(dataset) if dataset else None
                if want and not set(want) <= tidy.folds:
                    problems.append("folds "+str(sorted(tidy.folds))+" of "+str(want))
                if not tidy.pooled:
                    problems.append("no pooled out-of-fold rows")
        status = "PARTIAL" if problems else "STALE" if stale else "DONE"
        return status, problems + stale, tidy


def describe(tidy):
    """The ``--long`` note for a tidy file."""
    kind = ("%d-fold" % len(tidy.folds)+(" + pooled" if tidy.pooled else "")
            if tidy.folds else "single split")
    skipped = ", ".join(k+" "+str(v) for k, v in tidy.skipped.most_common())
    return (kind+", "+str(len(tidy.cells))+" cells ("+str(tidy.ok_cells)+" fitted), written "
            +day(tidy.mtime)+(", skipped rows: "+skipped if skipped else ""))


def file_unit(r, key, label, files, group="extra", hint="", **kwargs):
    """A unit whose output is simply a list of files that must all exist.

    ``hint`` says where a missing output comes from when it has no job of its
    own to submit (a log the measure scripts append to, a notebook's table).
    """
    present = [f for f in files if r.exists(f)]
    status = "DONE" if len(present) == len(files) else "PARTIAL" if present else "MISSING"
    detail, note = "", ""
    if status == "MISSING":
        detail = hint
    elif status == "DONE":
        note = "written "+day(max(os.path.getmtime(r.path(f)) for f in files))
    elif status == "PARTIAL":
        absent = [f for f in files if f not in present]
        detail = str(len(present))+"/"+str(len(files))+" files; missing "+", ".join(absent[:3])+(
            " ..." if len(absent) > 3 else "")
    return r.add(Unit(key=key, label=label, group=group, status=status, detail=detail,
                      note=note, **kwargs))


# ---------------------------------------------------------------------------
# IBM datasets
# ---------------------------------------------------------------------------

def is_score_only(dataset):
    """A Louvain-sweep arm (``_res<r>``, ``_nolouvain``): the pure score is compared, no model fitted.

    Hub removal is the other pre-processing arm and keeps its tree plan, and a
    bank view carries its own bank token, so neither counts.
    """
    return strip_resolution(dataset) != dataset and parse_hubs(dataset) is None


def big(dataset):
    return " --time=16:00:00 --mem=200g" if dataset.startswith("LI-Large") else ""


def measure_defect(path):
    """Why a directed measures file is stale, or ``None`` if it is current.

    Two ways: the file predates the block-size columns altogether, or it holds
    the values from before commit c5fba86, where a node with no level-2
    neighbours was handed a full block-12 density.
    """
    if "size_12" not in pd.read_csv(path, nrows=0).columns:
        return "no block-size columns (an older format the scores cannot be built from)"
    frame = pd.read_csv(path, usecols=["size_12", "measure_12"])
    bad = int(((frame["size_12"] == 0) & (frame["measure_12"] == 1)).sum())
    if bad:
        return f"{bad:,} of {len(frame):,} rows have {PRE_FIX}: computed before c5fba86"
    return None


def add_measures(r):
    """Stage 1: ``<dataset>_GARGAML_<direction>.csv`` for every IBM dataset name."""
    for i, dataset in enumerate(r.plan.ibm):
        for direction in DIRECTIONS:
            name = dataset+"_GARGAML_"+direction+".csv"
            col = "meas-D" if direction == "directed" else "meas-U"
            status, detail, note = "MISSING", "", ""
            if r.exists(name):
                status, note = "DONE", "written "+day(os.path.getmtime(r.path(name)))
                if direction == "directed" and r.check_values:
                    try:
                        defect = measure_defect(r.path(name))
                    except Exception as exc:  # noqa: BLE001
                        status, detail = "PARTIAL", "unreadable: "+str(exc)
                    else:
                        if defect:
                            status, detail = "STALE", defect+"; rerun with GARGAML_FORCE=1"
            cmd = ("slurm/submit.sh slurm/measures_ibm_"+("dir" if direction == "directed" else "undir")
                   +".slurm "+dataset+" --array="+str(i)+big(dataset))
            if status == "STALE":
                cmd = "GARGAML_FORCE=1 "+cmd
            r.add(Unit(key="meas:"+dataset+":"+direction, label="measures "+direction+" "+dataset,
                       group="ibm", status=status, detail=detail, note=note, cmd=cmd,
                       optional=dataset in OPTIONAL_DATASETS, cols={col: CELL[status]},
                       family="measures_ibm_"+("dir" if direction == "directed" else "undir"),
                       dataset=dataset, indices=frozenset([i])))


def add_folds(r):
    """The fold partition the tree job writes; GraphSAGE and FlowScope read it."""
    for i, dataset in enumerate(r.plan.ibm):
        if is_score_only(dataset):
            continue                  # no model, so no partition to share
        ids = r.folds(dataset)
        status = "DONE" if ids else "MISSING"
        r.add(Unit(key="folds:"+dataset, label="fold partition "+dataset, group="ibm",
                   status=status, note=("folds "+str(ids)) if ids else "",
                   needs=("meas:"+dataset+":directed", "meas:"+dataset+":undirected"),
                   optional=dataset in OPTIONAL_DATASETS, quiet=True,
                   cols={"folds": str(len(ids))+"f" if ids else "miss"},
                   family="tree", dataset=dataset, indices=frozenset([i])))


def tree_sections(direction):
    """Feature configs the tree job writes under ``direction``.

    A direction-free config (topology) runs once, in the first pass, which is
    the undirected one -- see the loop at the bottom of gargaml_tree.py.
    """
    return [c for c in FEATURE_CONFIGS if direction == "undirected" or not is_direction_free(c)]


def add_tree(r):
    for i, dataset in enumerate(r.plan.ibm):
        if is_score_only(dataset):
            continue
        statuses, reasons, notes, cols, taints = [], {}, [], {}, {}
        for direction in DIRECTIONS:
            col = "tree-D" if direction == "directed" else "tree-U"
            sections = tree_sections(direction)
            done = 0
            for config in sections:
                name = dataset+"_"+direction+config_suffix(config)+"_metrics.csv"
                status, why, tidy = r.section(name, r.plan.targets, dataset=dataset)
                statuses.append(status)
                done += status == "DONE"
                if status not in ("DONE", "MISSING"):
                    for reason in why:
                        reasons.setdefault(reason, []).append(direction[0].upper()+"-"+config)
                if tidy:
                    notes.append(direction+" "+config+": "+describe(tidy))
            cols[col] = ("done" if done == len(sections) else "miss" if not done
                         else str(done)+"/"+str(len(sections)))
            if "STALE" in statuses[-len(sections):]:
                cols[col] = "STALE"
            if direction == "directed":
                taints[col] = ("meas:"+dataset+":directed",)
        status = combine(statuses)
        r.add(Unit(key="tree:"+dataset, label="tree/boost "+dataset, group="ibm", status=status,
                   detail="; ".join(reason+" ("+("all sections" if len(ids) == len(statuses) else ", ".join(ids))+")"
                                    for reason, ids in reasons.items()),
                   note="\n".join(notes), needs=("meas:"+dataset+":directed", "meas:"+dataset+":undirected"),
                   taints=taints,
                   cmd="slurm/submit.sh slurm/tree.slurm "+dataset+" --array="+str(i)
                       +(" --cpus-per-task=36"+big(dataset) if dataset.startswith("LI-Large") else ""),
                   optional=dataset in OPTIONAL_DATASETS, cols=cols, family="tree", dataset=dataset,
                   indices=frozenset([i])))


def add_base_scores(r):
    """The base GARG-AML score through the shared metrics, per direction.

    The core datasets are scored per fold and pooled. A Louvain-sweep arm has
    no fold partition and fits nothing, so it gets full-population metrics
    only, and one job runs all of them (``SWEEP_JOB``).
    """
    arms = [d for d in r.plan.ibm if is_score_only(d) and d not in OPTIONAL_DATASETS]
    for dataset in CORE_DATASETS + arms:
        core = dataset in CORE_DATASETS
        statuses, problems, notes, cols = [], [], [], {}
        for direction in DIRECTIONS:
            status, why, tidy = r.section(dataset+"_"+direction+"_base_metrics.csv", r.plan.targets,
                                          dataset=dataset if core else None, cv=core)
            statuses.append(status)
            cols["base-D" if direction == "directed" else "base-U"] = CELL[status]
            if status != "DONE":
                problems.append(direction+": "+"; ".join(why))
            if tidy:
                notes.append(direction+": "+describe(tidy))
        needs = ("meas:"+dataset+":directed", "meas:"+dataset+":undirected")
        r.add(Unit(key="base:"+dataset, label="base score "+dataset, group="ibm",
                   status=combine(statuses), detail="; ".join(problems) if combine(statuses) != "DONE" else "",
                   note="\n".join(notes), needs=needs+(("folds:"+dataset,) if core else ()),
                   taints={"base-D": ("meas:"+dataset+":directed",)},
                   cmd=("slurm/submit.sh slurm/distribution_scores.slurm "+(dataset+big(dataset) if core else SWEEP_JOB)),
                   cols=cols, family="distribution_scores", dataset=dataset,
                   groups=() if core else (SWEEP_JOB,)))


def add_isolation_forest(r):
    dataset = IF_DATASET
    status, why, tidy = r.section(dataset+"_directed_if_metrics.csv", r.plan.targets, cv=False)
    r.add(Unit(key="if:"+dataset, label="isolation forest "+dataset, group="ibm", status=status,
               detail="; ".join(why) if status != "DONE" else "",
               note=(describe(tidy)+" (a single split by design)") if tidy else "",
               needs=("meas:"+dataset+":directed",), taints={"IF": ("meas:"+dataset+":directed",)},
               cmd="sbatch slurm/if.slurm",   # the dataset is hardcoded in gargaml_IF.py
               cols={"IF": CELL[status]},
               family="if", dataset=dataset))


def graphsage_parts(r, dataset, targets):
    """Targets whose per-target job wrote into ``graphsage_parts/``, merged or not."""
    tags = {t.replace(" ", "-"): t for t in targets}
    found = set()
    root = r.path("graphsage_parts")
    if os.path.isdir(root):
        for entry in os.listdir(root):
            if entry.startswith(dataset+"_") and entry[len(dataset)+1:] in tags:
                if os.path.isfile(os.path.join(root, entry, dataset+"_graphsage_tidy.csv")):
                    found.add(tags[entry[len(dataset)+1:]])
    return found


def add_graphsage(r):
    for dataset, setting in r.plan.graphsage.items():
        cut_offs, targets = setting["cut_offs"], setting["targets"]

        files = [dataset+"_graphsage_labels.pkl", dataset+"_graphsage_structure.pt",
                 dataset+"_graphsage_prep.csv"] + [
                 dataset+"_graphsage_x_"+config+".pt" for config in r.plan.graphsage_suffix]
        file_unit(r, "gsprep:"+dataset, "GraphSAGE caches "+dataset, files, group="ibm",
                  cmd="slurm/submit.sh slurm/graphsage_prep.slurm "+dataset
                      +(" --time=06:00:00 --mem=200g" if dataset.startswith("LI-Large") else ""),
                  family="graphsage_prep", dataset=dataset)

        in_parts = graphsage_parts(r, dataset, targets)
        statuses, counts, problems, notes, missing = [], {}, [], [], set()
        for config, suffix in r.plan.graphsage_suffix.items():
            name = dataset+"_undirected"+suffix+"_metrics.csv"
            status, why, tidy = r.section(name, targets, cut_offs=cut_offs, dataset=dataset)
            done = set()
            if tidy:
                # A file with no 0.0 rows is judged on the cut-offs it has; the
                # missing 0.0 is already reported as stale, once.
                have = [c for c in cut_offs if c != 0.0 or 0.0 in tidy.cutoffs]
                done = {t for t in targets if all((float(c), t) in tidy.cells for c in have)}
                notes.append(config+": "+describe(tidy))
                # Per-target jobs are merged into one file, so an incomplete sweep
                # is reported as targets, not as the cell count `section` found.
                why = [w for w in why if "cells" not in w]
                if len(done) < len(targets):
                    status = "PARTIAL"
            counts[config] = len(done)
            missing |= set(targets) - done
            # The attributes config dropped its timing features on 2026-10-06
            # (hindsight: an account's span is known only after the fact), so a
            # schema that still lists them marks every number in the run.
            schema = r.path(dataset+"_undirected"+suffix+"_feature_schema.csv")
            if tidy and os.path.isfile(schema):
                try:
                    timing = "timing" in set(pd.read_csv(schema)["kind"])
                except Exception:  # noqa: BLE001
                    timing = False
                if timing:
                    why.append(config+" schema still lists the timing features (predates 2026-10-06): rerun")
                    status = "STALE" if status == "DONE" else status
            statuses.append(status)
            problems += [w if w.startswith(config) else config+": "+w for w in why]
        status = combine(statuses)

        unmerged = missing & in_parts
        if status not in ("DONE", "MISSING"):
            progress = ", ".join(config+" "+str(n)+"/"+str(len(targets)) for config, n in counts.items())
            if missing:
                problems.insert(0, progress+" targets complete; unfinished: "+", ".join(t for t in targets if t in missing))
        if unmerged:
            problems.append(str(len(unmerged))+" finished target(s) sit in graphsage_parts/, not merged")
        stale = any("timing features" in p for p in problems)
        cell = ("done" if status == "DONE" else "miss" if status == "MISSING"
                else str(min(counts.values()))+"/"+str(len(targets))+("!" if stale else ""))
        cmd = ("GARGAML_DATASET="+dataset+" GARGAML_RESULTS_DIR="+r.dir+" GARGAML_MERGE_FROM="
               +r.dir+"/graphsage_parts python -u scripts/graphsage_baseline.py   # merge"
               if unmerged else "see slurm/README.md, 'GraphSAGE, one job per target'")
        r.add(Unit(key="sage:"+dataset, label="GraphSAGE "+dataset, group="ibm", status=status,
                   detail="; ".join(problems) if status != "DONE" else "", note="\n".join(notes),
                   needs=("folds:"+dataset, "gsprep:"+dataset), cmd=cmd, cols={"sage": cell},
                   family="graphsage", dataset=dataset))


# ---------------------------------------------------------------------------
# Synthetic grid
# ---------------------------------------------------------------------------

def tiers(plan):
    """``{n_nodes: [(index, name), ...]}``; the index is the array task id."""
    grouped = {}
    for i, name in enumerate(plan.synth):
        grouped.setdefault(int(name.split("_")[2]), []).append((i, name))
    return dict(sorted(grouped.items()))


def add_synthetic(r):
    plan = r.plan
    for n_nodes, members in tiers(plan).items():
        for direction in DIRECTIONS:
            token = "directed" if direction == "directed" else "undirected_parallel"
            present, stale, absent = [], [], []
            for i, name in members:
                path = r.path(name+"_GARGAML_"+token+".csv")
                if not (os.path.isfile(path) and os.path.getsize(path) > 0):
                    absent.append(i)
                    continue
                present.append(i)
                if direction == "directed" and r.check_values:
                    try:
                        if measure_defect(path):
                            stale.append(i)
                    except Exception:  # noqa: BLE001 - an unreadable file is a missing one
                        absent.append(i)
                        present.remove(i)
            status = ("MISSING" if not present else "PARTIAL" if absent else
                      "STALE" if stale else "DONE")
            time_, mem = SYNTH_COST.get(n_nodes, ("24:00:00", "64g"))
            script = "slurm/measures_synth_"+("dir" if direction == "directed" else "undir")+".slurm"
            lines = []
            if absent:
                lines.append("sbatch --array="+compress(absent)+" --time="+time_+" --mem="+mem+" "+script)
            if stale:
                lines.append("GARGAML_FORCE=1 sbatch --array="+compress(stale)+" --time="+time_
                             +" --mem="+mem+" "+script)
            bits = [str(len(present))+"/"+str(len(members))+" datasets"]
            if stale:
                bits.append(str(len(stale))+" stale (pre-c5fba86 values or an older format)")
            r.add(Unit(key="meas_synth:"+str(n_nodes)+":"+direction,
                       label="synthetic measures "+direction+", "+format(n_nodes, ",")+"-node",
                       group="synth", status=status,
                       detail=", ".join(bits) if status != "DONE" else "",
                       note=", ".join(bits), cmd="\n".join(lines),
                       family="measures_synth_"+("dir" if direction == "directed" else "undir"),
                       indices=frozenset(absent+stale)))

    small = min(tiers(plan))
    directed_keys = tuple("meas_synth:"+str(n)+":directed" for n in tiers(plan))
    variants = [("3", [n for n in plan.synth if n.endswith("_3")], False),
                ("5", [n for n in plan.synth if n.endswith("_5")], False),
                ("full", list(plan.synth), True)]
    for variant, expected, optional in variants:
        for direction in DIRECTIONS:
            name = "synthetic_tree_"+str(direction == "directed")+"_"+variant+".csv"
            status, detail = "MISSING", ""
            if r.exists(name):
                columns = set(pd.read_csv(r.path(name), index_col=0, nrows=0).columns)
                have = len(columns & set(expected))
                status = "DONE" if have == len(expected) else "PARTIAL"
                detail = "" if status == "DONE" else str(have)+"/"+str(len(expected))+" datasets"
            r.add(Unit(key="tree_synth:"+variant+"_"+direction,
                       label="synthetic tree/boost ("+("all patterns" if variant == "full" else variant+" patterns")+"), "+direction,
                       group="synth", status=status, detail=detail,
                       needs=("meas_synth:"+str(small)+":"+direction,),
                       taints={"": directed_keys} if direction == "directed" else {},
                       cmd="slurm/submit.sh slurm/tree_synth.slurm "+("base" if variant == "full" else variant)+"_"+direction,
                       optional=optional, family="tree_synth", dataset=("base" if variant == "full" else variant)+"_"+direction))
    for direction in DIRECTIONS:
        name = "synthetic_score_"+direction+"_supervised.csv"
        status, detail = "MISSING", ""
        if r.exists(name):
            columns = set(pd.read_csv(r.path(name), index_col=0, nrows=0).columns)
            have = len(columns & set(plan.synth))
            status = "DONE" if have == len(plan.synth) else "PARTIAL"
            detail = "" if status == "DONE" else str(have)+"/"+str(len(plan.synth))+" datasets"
        r.add(Unit(key="base_synth:"+direction, label="synthetic base score, "+direction, group="synth",
                   status=status, detail=detail, needs=("meas_synth:"+str(small)+":"+direction,),
                   taints={"": directed_keys} if direction == "directed" else {},
                   cmd="slurm/submit.sh slurm/distribution_scores.slurm synthetic",
                   family="distribution_scores", dataset="synthetic"))


# ---------------------------------------------------------------------------
# Appendices, diagnostics, reporting
# ---------------------------------------------------------------------------

def add_extras(r):
    plan = r.plan
    for dataset in CORE_DATASETS:
        file_unit(r, "labels:"+dataset, "label tables "+dataset,
                  [dataset+"_imbalance_"+d+"_combined.csv" for d in DIRECTIONS],
                  cmd="slurm/submit.sh slurm/label_distribution.slurm "+dataset+big(dataset),
                  family="label_distribution", dataset=dataset)
        file_unit(r, "psplit:"+dataset, "pattern splitting "+dataset,
                  [dataset+"_pattern_splitting.csv", dataset+"_pattern_splitting_summary.csv"],
                  cmd="slurm/submit.sh slurm/pattern_splitting.slurm "+dataset+big(dataset),
                  family="pattern_splitting", dataset=dataset)
        file_unit(r, "flowscope:"+dataset, "FlowScope metrics "+dataset,
                  [dataset+"_directed_flowscope_metrics.csv"], needs=("folds:"+dataset,),
                  cmd="slurm/submit.sh slurm/flowscope_evaluate.slurm "+dataset
                      +(" --time=06:00:00 --mem=200g" if dataset == "LI-Large" else " --mem=64g")
                      +"   # needs the FlowScope fork",
                  family="flowscope_evaluate", dataset=dataset)
    for spec in plan.institutions:
        view = plan.view_base+"_bank"+spec
        file_unit(r, "po:"+spec, "partial observability, bank "+spec,
                  [view+"_partial_observability_accounts.csv", view+"_partial_observability_metrics.csv"],
                  cmd="slurm/submit.sh slurm/partial_obs.slurm "+spec, family="partial_obs", dataset=spec)

    small = [n for n in plan.synth if "_100_" in n]
    files = [n+suffix+".csv" for n in small
             for suffix in ("_directed_diagnosis", "_directed_diagnosis_summary", "_directed_diagnosis_metrics")]
    files += ["directed_diagnosis_summary.csv", "directed_diagnosis_metrics.csv"]
    file_unit(r, "diagnosis", "directed-vs-undirected diagnosis ("+str(len(small))+" 100-node datasets)", files,
              cmd="sbatch slurm/directed_diagnosis.slurm", family="directed_diagnosis")

    file_unit(r, "file:louvain_severance", "Louvain edge severance log", ["louvain_severance.csv"],
              hint="appended by the measure scripts", quiet=True)
    file_unit(r, "file:preprocessing_severance", "pre-processing severance table",
              ["preprocessing_severance.csv"], hint="run notebooks/LouvainEdgeSeverance.ipynb", quiet=True)
    file_unit(r, "file:pattern_splitting_summary", "pooled pattern-splitting summary",
              ["pattern_splitting_summary.csv"], hint="written by slurm/pattern_splitting.slurm", quiet=True)

    tables = [f for f in os.listdir(r.dir) if f.startswith("table_") and f.endswith(".tex")] \
        if os.path.isdir(r.dir) else []
    needs = ("tree:HI-Small",)
    status = "DONE" if tables and r.exists("table_coverage.csv") else "PARTIAL" if tables or r.exists("table_coverage.csv") else "MISSING"
    r.add(Unit(key="tables", label="result tables (build_tables.py)", group="extra", status=status,
               detail=(str(len(tables))+" .tex files") if status != "DONE" else "", note=str(len(tables))+" .tex files",
               needs=needs, cmd="slurm/submit.sh slurm/collect.slurm all", family="collect", dataset="all"))
    file_unit(r, "timing", "timing_all.csv", ["timing_all.csv"], needs=needs,
              cmd="slurm/submit.sh slurm/collect.slurm all", family="collect", dataset="all")
    for label, files in [("CD diagrams and Friedman tests", ["CD_ROC_full.pdf", "CD_PR_full.pdf", "friedman_results.csv"]),
                         ("runtime figures", ["time_boxplot_norm.pdf", "time_boxplot_ibm.pdf"])]:
        file_unit(r, "notebook:"+label, label+" (notebook)", files,
                  cmd="run notebooks/"+("VisualisationResults" if "CD" in label else "VisualisationRunTime")+".ipynb")


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    name: str
    state: str
    elapsed: str
    reason: str
    family: str = None
    dataset: str = None
    array: set = None


def slurm_families():
    """``(default job name -> script basename, basenames)`` from the .slurm headers."""
    names, bases = {}, []
    if not os.path.isdir("slurm"):
        return names, bases
    for entry in sorted(os.listdir("slurm")):
        if not entry.endswith(".slurm"):
            continue
        base = entry[:-len(".slurm")]
        bases.append(base)
        with open(os.path.join("slurm", entry)) as fh:
            for line in fh:
                match = re.match(r"#SBATCH\s+--job-name[= ]\s*(\S+)", line)
                if match:
                    names[match.group(1)] = base
    return names, bases


def parse_array(job_id):
    """Task ids of an array job id (``123_[44-65%4]``, ``123_7``), else ``None``."""
    match = re.search(r"_\[?([\d,\-]+)(?:%\d+)?\]?$", job_id)
    if not match:
        return None
    indices = set()
    for part in match.group(1).split(","):
        if "-" in part:
            low, high = part.split("-")
            indices.update(range(int(low), int(high)+1))
        elif part:
            indices.add(int(part))
    return indices


def parse_queue(text, names, bases):
    """Jobs out of ``squeue -h -o '%i|%j|%T|%M|%R'``; other lines are skipped."""
    jobs = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5:
            continue          # the "CLUSTER: wice" banner of a multi-cluster squeue
        job = Job(*[p.strip() for p in parts], array=parse_array(parts[0].strip()))
        if job.name in names:
            job.family = names[job.name]
        else:
            for base in sorted(bases, key=len, reverse=True):
                if job.name == base:
                    job.family = base
                    break
                if job.name.startswith(base+"_"):
                    job.family, job.dataset = base, job.name[len(base)+1:]
                    break
        jobs.append(job)
    return jobs


def queue_jobs(cluster):
    """``(jobs, note)``; ``jobs`` is ``None`` when the queue cannot be read."""
    if shutil.which("squeue") is None:
        return None, "squeue not found (not on the cluster): queue not checked"
    command = ["squeue", "-h", "-u", os.environ.get("USER") or getpass.getuser(),
               "-o", "%i|%j|%T|%M|%R"]
    if cluster:
        command[1:1] = ["-M", cluster]
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "squeue failed ("+str(exc)+"): queue not checked"
    if out.returncode != 0:
        return None, "squeue failed ("+out.stderr.strip()[:120]+"): queue not checked"
    names, bases = slurm_families()
    return parse_queue(out.stdout, names, bases), ""


def job_matches(job, unit):
    """Does this queued job belong to this experiment?

    A job submitted through ``slurm/submit.sh`` is named ``<script>_<dataset>``
    and matches on both. A bare ``sbatch`` keeps the script's default name, so
    only an array job can be placed, by its task ids against the unit's own.
    """
    if not job.family or job.family != unit.family:
        return False
    if job.dataset is not None:
        return job.dataset == unit.dataset or job.dataset in unit.groups
    if job.array is not None:
        return bool(unit.indices & job.array)
    return True


# ---------------------------------------------------------------------------
# Resolving states
# ---------------------------------------------------------------------------

def resolve(units, jobs):
    """Turn each unit's output status into the state the report shows."""
    by_key = {u.key: u for u in units}

    for u in units:                                   # results built on stale inputs
        if u.status != "DONE":
            continue
        tainted = []
        for col, keys in u.taints.items():
            bad = [by_key[k].label for k in keys if by_key[k].status == "STALE"]
            if bad:
                tainted.append(col)
                if col in u.cols:
                    u.cols[col] = "STALE"
                u.detail = (u.detail+"; " if u.detail else "")+"built on stale "+", ".join(bad[:2])+(
                    " ..." if len(bad) > 2 else "")
        if tainted:
            u.status = "STALE"

    for u in units:
        u.jobs = tuple(j for j in (jobs or []) if job_matches(j, u)) if u.status not in ("DONE", "NA") else ()
        if u.status != "MISSING" or (u.quiet and not u.needs):
            u.state = u.status
        else:
            u.waiting_on = tuple(by_key[k].label for k in u.needs
                                 if by_key[k].status not in ("DONE", "STALE", "NA"))
            u.state = "BLOCKED" if u.waiting_on else "READY"
        if u.jobs:
            u.state = "RUNNING" if any(j.state in ("RUNNING", "COMPLETING") for j in u.jobs) else "QUEUED"
        u.cols = {col: STATE_WORD[u.state] if text == "miss" else text for col, text in u.cols.items()}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLOURS = {"done": "32", "STALE": "35", "ready": "36", "wait": "90", "RUN": "34;1",
           "PEND": "34", "-": "90", "miss": "90", "part": "33"}


def paint(text, enabled):
    """Wrap a (padded) cell in the ANSI colour its word maps to."""
    word = text.strip()
    if not enabled or not word:
        return text
    if word in COLOURS:
        code = COLOURS[word]
    elif "STALE" in word or word.endswith("!"):
        code = "35"
    elif re.fullmatch(r"\d+/\d+", word):
        code = "33"
    elif re.fullmatch(r"\d+f", word):
        code = "32"
    else:
        return text
    return "\033["+code+"m"+text+"\033[0m"


def table(headers, rows, colour=False, painted=()):
    """Left-aligned text table; the columns in ``painted`` are colourised."""
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def line(cells, paint_cells):
        parts = []
        for i, (text, width) in enumerate(zip(cells, widths)):
            padded = str(text).ljust(width)
            parts.append(paint(padded, colour) if paint_cells and i in painted else padded)
        return "  ".join(parts).rstrip()

    return "\n".join([line(headers, False)]+[line(row, True) for row in rows])


def matrix_rows(r):
    cells = {}
    for u in r.units:
        if u.group == "ibm":
            for col, text in u.cols.items():
                cells[(u.dataset, col)] = text
    return [[dataset+(" (opt)" if dataset in OPTIONAL_DATASETS else "")]
            + [cells.get((dataset, col), "-") for col in MATRIX] for dataset in r.plan.ibm]


NEXT_STEPS = [("RUNNING", "Running now"), ("QUEUED", "Pending in the queue"),
              ("PARTIAL", "Incomplete: finish these"), ("STALE", "Stale: rerun to replace"),
              ("READY", "Ready to submit (inputs are on disk)"), ("BLOCKED", "Waiting on an earlier step")]


def queue_line(r, jobs):
    """One line on the queue: how many of the user's jobs there are, and the unplaced ones."""
    placed = {j.job_id for u in r.units for j in u.jobs}
    stray = [j for j in jobs if j.job_id not in placed]
    line = "Queue: "+str(len(jobs))+" job(s) of yours, "+str(len(jobs)-len(stray))+" matched to an unfinished experiment"
    if stray:
        line += "; not matched: "+", ".join(j.name+" ("+j.job_id+" "+j.state.lower()+")" for j in stray[:6])
        line += " ..." if len(stray) > 6 else ""
    return line


def render(r, jobs, queue_note, long=False, colour=False):
    out = []
    required = [u for u in r.units if not (u.optional or u.quiet)]
    tally = Counter(u.state for u in required)
    out.append("=== GARG-AML experiment status: "+r.dir+"/  ("+time.strftime("%Y-%m-%d %H:%M")+") ===")
    out.append("   ".join(word+" "+str(tally[state]) for state, word in
                          [("DONE", "done"), ("PARTIAL", "partial"), ("STALE", "stale"), ("RUNNING", "running"),
                           ("QUEUED", "queued"), ("READY", "ready"), ("BLOCKED", "waiting")] if tally[state])
               +"   of "+str(len(required))+" experiments (optional ones not counted)")
    if queue_note:
        out.append(queue_note)
    elif jobs:
        out.append(queue_line(r, jobs))

    out.append("\n-- IBM datasets --")
    out.append(table(["dataset"]+MATRIX, matrix_rows(r), colour, painted=range(1, len(MATRIX)+1)))
    out.append("   D = directed, U = undirected, Nf = N-fold partition on disk, 2/4 = sections or targets done,\n"
               "   ! = a GraphSAGE attributes run that still carries the timing features, - = not in the plan")

    for group, title in (("synth", "Synthetic grid ("+str(len(r.plan.synth))+" datasets)"),
                         ("extra", "Appendices, diagnostics, reporting")):
        units = [u for u in r.units if u.group == group]
        rows = [[u.label+(" (opt)" if u.optional else ""), STATE_WORD[u.state],
                 u.detail or (u.note if long else "")] for u in units]
        out.append("\n-- "+title+" --")
        out.append(table(["experiment", "state", "detail"], rows, colour, painted=(1,)))

    if long:
        out.append("\n-- Notes, IBM experiments --")
        for u in (u for u in r.units if u.group == "ibm" and (u.note or u.detail)):
            out.append(u.label+" ["+STATE_WORD[u.state]+"]")
            out.extend("    "+text for text in u.note.split("\n") if text)
            if u.detail:
                out.append("    ! "+u.detail)

    out.append("\n-- Next steps --")
    printed = False
    for state, title in NEXT_STEPS:
        group = [u for u in r.units if u.state == state and not (u.optional or u.quiet)]
        if not group:
            continue
        printed = True
        out.append("\n"+title+":")
        seen = set()
        for u in group:
            line = "  "+u.label
            if u.jobs:
                line += "   (job "+", ".join(j.job_id+" "+j.state.lower() for j in u.jobs[:3])+")"
            if state == "BLOCKED":
                line += "   <- waiting on: "+"; ".join(u.waiting_on)
            elif u.detail:
                line += "   - "+u.detail
            out.append(line)
            if state in ("READY", "PARTIAL", "STALE") and u.cmd:
                out.extend("      "+(cmd_line if cmd_line not in seen else "(same job as above)")
                           for cmd_line in u.cmd.split("\n"))
                seen.update(u.cmd.split("\n"))
    optional = [u for u in r.units if u.optional and u.state not in ("DONE", "NA")]
    if optional:
        printed = True
        names = sorted({u.dataset if u.group == "ibm" else u.label for u in optional})
        out.append("\nOptional, expected not to run (slurm/README.md, Known gaps): "+", ".join(names))
    if not printed:
        out.append("Nothing outstanding.")
    return "\n".join(out)


def to_frame(r):
    return pd.DataFrame([{
        "results_dir": r.dir, "key": u.key, "label": u.label, "group": u.group, "dataset": u.dataset,
        "state": u.state, "optional": u.optional, "detail": u.detail,
        "waiting_on": "; ".join(u.waiting_on), "jobs": ", ".join(j.job_id for j in u.jobs), "cmd": u.cmd,
    } for u in r.units])


def build(results_dir, plan, check_values):
    r = Report(results_dir, plan, check_values=check_values)
    add_measures(r)
    add_folds(r)
    add_tree(r)
    add_base_scores(r)
    add_isolation_forest(r)
    add_graphsage(r)
    add_synthetic(r)
    add_extras(r)
    return r


def parse_args():
    parser = argparse.ArgumentParser(description="Which experiments have run, and which cannot run yet.")
    parser.add_argument("--dir", nargs="+", default=None,
                        help="results directories to report, one report each "
                             "(default: $GARGAML_RESULTS_DIR, else results)")
    parser.add_argument("--long", action="store_true", help="notes for every experiment, skipped-row counts included")
    parser.add_argument("--no-values", action="store_true",
                        help="skip the value-level checks (rows of the directed measures); faster, "
                             "but pre-fix measures then read as done")
    parser.add_argument("--no-queue", action="store_true", help="do not call squeue")
    parser.add_argument("--cluster", default=os.environ.get("GARGAML_SLURM_CLUSTER", "wice"),
                        help="squeue -M argument (default: wice; empty for a single-cluster site)")
    parser.add_argument("--csv", default=None, help="write one row per experiment to this file")
    return parser.parse_args()


def main():
    args = parse_args()
    plan = load_plan()
    jobs, note = ([], "") if args.no_queue else queue_jobs(args.cluster)
    frames = []
    for results_dir in args.dir or [resolve_results_dir()]:
        if not os.path.isdir(results_dir):
            print("No directory "+results_dir+"/ -- nothing to report.")
            continue
        report = build(results_dir, plan, check_values=not args.no_values)
        resolve(report.units, jobs)
        print(render(report, jobs, note, long=args.long, colour=sys.stdout.isatty() and not os.environ.get("NO_COLOR")))
        print()
        frames.append(to_frame(report))
    if args.csv and frames:
        pd.concat(frames, ignore_index=True).to_csv(args.csv, index=False)
        print("status -> "+args.csv)


if __name__ == "__main__":
    main()
