"""
Strip Jupyter notebook outputs before they reach git.

A repository hygiene tool rather than a pipeline stage: notebook outputs
are bulky, mostly base64 figure data, and hold nothing a re-run does not
regenerate.

Standard library only, deliberately: this runs as a git *clean filter*,
so it executes on every ``git add`` and ``git status`` of a notebook. A
filter that depends on a package someone forgot to install fails at the
least convenient moment, and a missing filter silently puts the outputs
back. ``nbstripout`` is the usual tool for this job and does strictly
more; it is not used here only to keep the filter dependency-free.

Formatting is preserved exactly. ``json.dumps(..., indent=1,
sort_keys=True, separators=(",", ": "))`` plus a trailing newline
reproduces nbformat's own writer byte for byte, so a stripped notebook
diffs only where the outputs were, with no reformatting churn.

Three ways to run it::

    git config filter.nbstrip.clean "python scripts/nbstrip.py --filter"
    python scripts/nbstrip.py notebooks/*.ipynb    # strip in place
    python scripts/nbstrip.py --check notebooks/   # CI/pre-commit gate

As a filter it is fail-safe: anything it cannot parse as a notebook is
passed through unchanged, because git replaces the file content with
whatever this writes to stdout and a crash mid-filter must not be able to
truncate a notebook.
"""

import json
import sys
from pathlib import Path

# Per-cell metadata that records *this* execution rather than the
# notebook's content: timings, fold/scroll state left over from the UI.
VOLATILE_CELL_METADATA = ("execution", "collapsed", "scrolled", "ExecuteTime")

# Notebook-level metadata holding ipywidgets state, which is both large
# and meaningless without the live kernel that produced it.
VOLATILE_NOTEBOOK_METADATA = ("widgets",)


def strip_notebook(nb):
    """Remove outputs and execution state from a parsed notebook.

    Mutates and returns ``nb``. Markdown cells are left alone; a code
    cell keeps its source and its own real metadata (tags, for instance)
    and loses only what a re-run would regenerate.
    """
    for key in VOLATILE_NOTEBOOK_METADATA:
        nb.get("metadata", {}).pop(key, None)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        for key in VOLATILE_CELL_METADATA:
            cell.get("metadata", {}).pop(key, None)

    return nb


def dump_notebook(nb):
    """Serialise exactly as nbformat does, so diffs stay minimal."""
    return json.dumps(
        nb, indent=1, sort_keys=True, ensure_ascii=False, separators=(",", ": ")
    ) + "\n"


def strip_text(text):
    """Strip a notebook given as text; return it unchanged if unparseable."""
    try:
        nb = json.loads(text)
    except ValueError:
        return text, False
    if not isinstance(nb, dict) or "cells" not in nb:
        return text, False
    return dump_notebook(strip_notebook(nb)), True


def run_filter():
    """git clean-filter mode: stdin -> stdout, never failing destructively."""
    text = sys.stdin.read()
    stripped, parsed = strip_text(text)
    if not parsed:
        print("nbstrip: not a notebook, passed through unchanged", file=sys.stderr)
    sys.stdout.write(stripped)
    return 0


def notebook_paths(arguments):
    """Expand the command line into notebook paths (directories recurse)."""
    paths = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            paths += sorted(path.rglob("*.ipynb"))
        elif path.suffix == ".ipynb":
            paths.append(path)
    return [p for p in paths if ".ipynb_checkpoints" not in p.parts]


def run_files(arguments, check_only):
    """Strip files in place, or report which ones would change."""
    paths = notebook_paths(arguments)
    if not paths:
        print("nbstrip: no notebooks found in "+" ".join(arguments), file=sys.stderr)
        return 1

    changed = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        stripped, parsed = strip_text(text)
        if not parsed:
            print("SKIP   "+str(path)+" (not a notebook)")
            continue
        if stripped == text:
            continue
        changed.append(path)
        # A notebook whose cells only carry execution counts can get a
        # byte or two *larger* (a number becomes ``null``), so report the
        # saving only when there is one rather than printing "-0 KB".
        saved = (len(text) - len(stripped)) / 1024
        detail = "-"+format(saved, ".0f")+" KB" if saved >= 0.5 else "execution state only"
        verb = "would strip" if check_only else "stripped"
        print(verb+" "+str(path)+"  ("+detail+")")
        if not check_only:
            path.write_text(stripped, encoding="utf-8")

    if not changed:
        print("nbstrip: "+str(len(paths))+" notebooks already clean")
    return 1 if (check_only and changed) else 0


def main(argv):
    arguments = [a for a in argv if not a.startswith("--")]
    if "--filter" in argv:
        return run_filter()
    if not arguments:
        print(__doc__.strip().split("Three ways")[-1], file=sys.stderr)
        return 1
    return run_files(arguments, check_only="--check" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
