"""Environment overrides for the run-scope constants in ``scripts/``.

Every runnable script configures its run with module-level constants --
``datasets``, ``N_FOLDS``, ``CONFIGS`` and so on. Editing them by hand is fine
locally and wrong on a cluster: a Slurm array task cannot edit the source it
shares with 65 of its siblings, and a run configured by an edit is not
reproducible from its log.

The constants stay where they are, as the defaults, and ``GARGAML_<NAME>``
overrides one for the duration of a job::

    DATASETS = env_override("datasets", DATASETS, as_list)
    N_FOLDS  = env_override("n_folds", 5, int)

Running a script bare therefore uses the defaults; the Slurm scripts set the
environment instead of patching the file.

``select_datasets`` turns ``SLURM_ARRAY_TASK_ID`` into a single entry of the
work list, so one array task runs one dataset. It fails loudly on an
out-of-range index rather than running nothing, since an array task that exits
0 having done no work is the failure mode hardest to notice afterwards.

``write_csv`` is here rather than in ``evaluation.py`` because stage 1 needs it
too and does not import that module. A plain ``df.to_csv(final_path)`` killed
on wall time leaves a truncated CSV at the real filename -- which downstream
readers parse happily as a short file, and which the skip-if-exists resume
(``GARGAML_FORCE``) would treat as finished work.
"""

import os
import sys


def env_override(key, default, cast=str):
    """``GARGAML_<KEY>``, parsed by ``cast``, or ``default`` when unset.

    An empty value counts as unset, so ``GARGAML_DATASETS=`` in a batch script
    does not silently blank a work list.
    """
    raw = os.environ.get("GARGAML_" + key.upper())
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw)
    except Exception as exc:                      # noqa: BLE001 - re-raised with context
        raise ValueError(
            "GARGAML_" + key.upper() + "=" + repr(raw) + " could not be parsed by "
            + getattr(cast, "__name__", str(cast)) + ": " + str(exc)
        ) from exc


def as_list(raw):
    """Comma-separated string -> list of stripped, non-empty strings."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def as_bool(raw):
    """``1/true/yes/on`` (any case) -> True; ``0/false/no/off`` -> False."""
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("expected a boolean (1/0, true/false, yes/no, on/off)")


def select_datasets(default_datasets):
    """The work list for this process, honouring the array-job environment.

    Resolution order, first match wins:

    1. ``GARGAML_DATASET`` -- one name, used verbatim. Not checked against
       ``default_datasets``: a dataset string is a tag in this repository
       ("HI-Small_res20", "HI-Small_bank012"), so a legitimate name need not
       appear in any hardcoded list.
    2. ``GARGAML_DATASETS`` -- a comma-separated work list, used verbatim.
    3. ``GARGAML_DATASET_INDEX``, else ``SLURM_ARRAY_TASK_ID`` -- an index into
       ``default_datasets``. Out of range is a hard error.
    4. Nothing set -- ``default_datasets`` unchanged.

    Always returns a list, so the caller's ``for dataset in datasets`` loop is
    untouched.
    """
    name = env_override("dataset", None)
    if name is not None:
        return [name]

    names = env_override("datasets", None, as_list)
    if names:
        return names

    raw_index = os.environ.get("GARGAML_DATASET_INDEX")
    if raw_index is None or raw_index.strip() == "":
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_index is None or raw_index.strip() == "":
        return list(default_datasets)

    try:
        index = int(raw_index)
    except ValueError:
        raise ValueError(
            "dataset index " + repr(raw_index) + " is not an integer "
            "(from GARGAML_DATASET_INDEX or SLURM_ARRAY_TASK_ID)"
        ) from None

    if not 0 <= index < len(default_datasets):
        raise IndexError(
            "dataset index " + str(index) + " is out of range: this script's work "
            "list holds " + str(len(default_datasets)) + " entries, so the valid "
            "array range is 0-" + str(len(default_datasets) - 1) + ". "
            "Check the --array spec against slurm/common.sh."
        )
    return [default_datasets[index]]


def resolve_results_dir(default="results"):
    """``GARGAML_RESULTS_DIR`` override, or *default* when unset.

    Every script builds its results and measures paths from this rather than
    from a literal "results/", so a full rerun can land in a fresh folder
    without touching an existing one.
    """
    return env_override("results_dir", default)


def echo_config(script, **knobs):
    """Print the resolved run configuration, so the .out file reconstructs it.

    Slurm identity is included when present, which is what pairs a job id
    with the settings it ran under.
    """
    parts = []
    for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_CPUS_PER_TASK", "SLURMD_NODENAME"):
        value = os.environ.get(key)
        if value:
            parts.append(key.replace("SLURM_", "").replace("SLURMD_", "").lower()
                         + "=" + value)
    for key, value in knobs.items():
        parts.append(key + "=" + repr(value))
    print("[config] " + script + ": " + " ".join(parts), flush=True)


def force_rerun():
    """``GARGAML_FORCE=1`` -- recompute even when the output is already on disk.

    The measure scripts overwrite in place, so the resume guard has to be
    escapable for a run whose point is to replace an existing file.
    """
    return env_override("force", False, as_bool)


def should_skip(out_path, dataset=""):
    """True when ``out_path`` already exists and no rerun was forced.

    Prints what it skipped, so a resubmitted array job's log says which tasks
    were no-ops rather than appearing to have done the work again.
    """
    if force_rerun() or not os.path.exists(out_path):
        return False
    label = dataset or os.path.basename(out_path)
    print("=== " + str(label) + ": already on disk, skipping (GARGAML_FORCE=1 to "
          "recompute) ===", flush=True)
    return True


def write_csv(df, path, **kwargs):
    """``df.to_csv`` via a temp file in the destination directory + ``os.replace``.

    ``os.replace`` is atomic within a filesystem, so a reader either sees the
    previous file or the complete new one, never a half-written one. The temp
    file must live in the destination directory for that guarantee to hold --
    a rename across filesystems is a copy.
    """
    kwargs.setdefault("index", False)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, "." + os.path.basename(path) + ".tmp" + str(os.getpid()))
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return path


def timing_path(dataset, direction, results_dir="results"):
    """Per-task timing file under ``results/timing/``.

    One file per task, because ``results/time_results_*.txt`` is opened in
    append mode and shared by three scripts -- a race under an array job,
    and unattributable afterwards since a row there is just ``<dataset>:
    <seconds>`` with no job, host, worker count or date. That file is left
    unchanged, so notebooks/VisualisationRunTime.ipynb still reads it fine.
    """
    directory = os.path.join(results_dir, "timing")
    os.makedirs(directory, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    stem = str(dataset) + "_" + str(direction) + "_" + job
    if task:
        stem += "_" + task
    return os.path.join(directory, stem + ".csv")


def log_timing(dataset, direction, elapsed, n_workers, results_dir="results"):
    """Record one dataset's wall time with enough identity to attribute it later."""
    import socket
    from datetime import datetime, timezone

    path = timing_path(dataset, direction, results_dir)
    row = {
        "dataset": dataset,
        "direction": direction,
        "seconds": round(float(elapsed), 2),
        "n_workers": n_workers,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "host": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    header = ",".join(row.keys())
    values = ",".join(str(v) for v in row.values())
    with open(path, "w") as handle:
        handle.write(header + "\n" + values + "\n")
    return path


if __name__ == "__main__":
    # Sanity check: `python src/utils/runtime.py` echoes what the environment
    # currently resolves to, without importing any of the pipeline.
    demo = ["alpha", "beta", "gamma"]
    print("default work list:", demo)
    try:
        print("resolved:", select_datasets(demo))
    except (IndexError, ValueError) as exc:
        print("resolution failed:", exc)
        sys.exit(1)
    echo_config("runtime.py self-test", force=force_rerun())
