"""
AutoAudit on the synthetic grid: ranking metrics from the orderings it saved.

The AutoAudit fork's AA-Smurf-Eval.py reduces each dataset to Precision, F1,
AUC-ROC and AUC-PR and keeps nothing else, so
results-aa/synthetic_autoaudit_combined_2.csv has no Precision@K. The
orderings behind it are still on disk -- AA-Smurf.py writes
results-2/order_<dataset><seed>.pkl and data/map_id_inv_<dataset><seed>.pkl --
so this re-scores them the way AA-Smurf-Eval.py does (rank score
(N - j) / N, 0 for a node AutoAudit never ranked, seeds 0-2 averaged) and adds
the ranking metrics through src/utils/evaluation.py. AutoAudit is not re-run.

Two populations per cell, in the layout the tree and base-score files use:

    'autoaudit'       the tree models' 30% test rows for that label, so P@10
                      is ranked over the same nodes as every GARG-AML model.
    'autoaudit_full'  every node -- the population of the published tuple, which
                      it reproduces (check it against the combined file before
                      trusting the rest).

A dataset without all three seeds' orderings is written as a gap (-1, read
back as NaN), never as zeros: the fork's combined file holds (0, 0, 0, 0) for
the datasets AutoAudit did not finish (all of 100,000 nodes and four 10,000-node
Watts-Strogatz ones), and a zero is a score, not an absence.

Precision and F1 are for the rule 'rank score > 0.5', i.e. the top half of
AutoAudit's ordering, as in AA-Smurf-Eval.py; the threshold-free metrics do not
depend on it.

    GARGAML_RESULTS_DIR=results-revision python scripts/autoaudit_synthetic.py

Run from the repository root. GARGAML_AUTOAUDIT_DIR points at the fork
(default: a sibling of GARG-AML, ../AutoAudit/AutoAudit). Writes
<results dir>/synthetic_autoaudit_ranking.csv.
"""
import os
import sys

DIR = "./"
os.chdir(DIR)
sys.path.append(DIR)

import pickle

import numpy as np
import pandas as pd

from scripts.gargaml_tree_synthetic import _legacy_metric_dict, holdout_test_index
from scripts.status import script_function
from src.utils.evaluation import evaluate_scores, nan_metrics, single_class_reason
from src.utils.runtime import echo_config, env_override, resolve_results_dir, write_csv

RESULTS_DIR = resolve_results_dir()
AUTOAUDIT_DIR = env_override("autoaudit_dir", "../AutoAudit/AutoAudit")

PATTERNS = ["laundering", "separate", "new_mules", "existing_mules"]

# AA-Smurf-Eval.py averages three random index initialisations, seeds 0-2. Some
# datasets have orderings for seeds 3-4 as well; they were never in the
# published numbers, so they are not in these.
SEEDS = range(3)

# Precision and F1 count the top half of AutoAudit's ordering as flagged.
RANK_THRESHOLD = 0.5


def seed_paths(dataset, seed):
    """The two pickles one (dataset, seed) ordering is rebuilt from."""
    return (os.path.join(AUTOAUDIT_DIR, "results-2", "order_"+dataset+str(seed)+".pkl"),
            os.path.join(AUTOAUDIT_DIR, "data", "map_id_inv_"+dataset+str(seed)+".pkl"))


def missing_files(dataset):
    """Every ordering pickle of ``dataset`` that is not on disk, for all seeds."""
    return [path for seed in SEEDS for path in seed_paths(dataset, seed)
            if not os.path.exists(path)]


def load_pickle(path):
    """``pickle.load`` that names the file when the read itself fails.

    A pickle that is on disk but cannot be read (a cloud-synced folder whose
    file was never downloaded fails with 'Operation timed out' from inside
    ``pickle.load``, naming nothing) stops the run: it is not a dataset
    AutoAudit lacks, so it must not become a gap.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise OSError("cannot read "+path+": "+str(exc)) from exc
    return pickle.loads(raw)


def rank_scores(dataset, seed, n_nodes):
    """AutoAudit's score for nodes 0..n_nodes-1: (N - j) / N by rank j, else 0.

    AA-Smurf-Eval.py's construction. A node is the row index of the label file,
    so the array is indexed by node id.
    """
    order_path, map_path = seed_paths(dataset, seed)
    order_nodes = load_pickle(order_path)
    map_id_inv = load_pickle(map_path)

    ranked = [map_id_inv[i] for i in order_nodes]
    score_by_node = {node: (len(ranked)-j)/len(ranked) for j, node in enumerate(ranked)}
    return pd.Series(np.arange(n_nodes)).map(score_by_node).fillna(0).to_numpy()


def evaluate_population(y, seed_scores):
    """Every metric on one population, averaged over the seeds' orderings.

    A population with one class only is a gap: AUC is undefined there, and the
    stratified split does not guard against it.
    """
    if single_class_reason(y) is not None:
        return nan_metrics()
    per_seed = [evaluate_scores(y, s, y_pred=(s > RANK_THRESHOLD).astype(int))
                for s in seed_scores]
    return {name: float(np.mean([m[name] for m in per_seed])) for name in per_seed[0]}


def autoaudit_cell(metrics):
    """A metric dict as a result cell: the tree files' key spelling, NaN as -1."""
    return _legacy_metric_dict({name: float(value) for name, value in metrics.items()})


def autoaudit_dataset(dataset):
    """``{pattern: {'autoaudit': cell, 'autoaudit_full': cell}}`` for one dataset."""
    labels = pd.read_csv("data/label_data_"+dataset+".csv")
    seed_scores = [rank_scores(dataset, seed, len(labels)) for seed in SEEDS]

    results = {}
    for pattern in PATTERNS:
        y = (labels[pattern]*1).to_numpy()
        # The rows the tree and base-score models are tested on. Passed the
        # label frame in node order, not the merge_labels frame: that one
        # takes its row order from the pandas outer-merge, which keeps the
        # measures file's node order up to pandas 2.1 and sorts by node from
        # 2.2, and the stored cluster results are the sorted order. Here the
        # label file already is that order, whatever pandas runs this.
        test = holdout_test_index(labels, pattern).to_numpy()

        results[pattern] = {
            "autoaudit": autoaudit_cell(evaluate_population(y[test], [s[test] for s in seed_scores])),
            "autoaudit_full": autoaudit_cell(evaluate_population(y, seed_scores)),
        }
    return results


def benchmark_synthetic():
    datasets = script_function("gargaml_directed_synth.py", "construct_datasets")()

    gap = autoaudit_cell(nan_metrics())
    gap_cell = {pattern: {"autoaudit": gap, "autoaudit_full": gap} for pattern in PATTERNS}

    results, skipped = {}, {}
    for dataset in datasets:
        missing = missing_files(dataset)
        if missing:
            skipped[dataset] = len(missing)
            results[dataset] = gap_cell
            continue
        print("====", dataset, "====", flush=True)
        results[dataset] = autoaudit_dataset(dataset)

    out_path = RESULTS_DIR+"/synthetic_autoaudit_ranking.csv"
    write_csv(pd.DataFrame(results), out_path, index=True)
    print("wrote "+out_path)
    print(f"{len(datasets)-len(skipped)} of {len(datasets)} datasets scored; "
          f"{len(skipped)} written as gaps (orderings not on disk under {AUTOAUDIT_DIR}):")
    for dataset in skipped:
        print("   "+dataset)


if __name__ == "__main__":
    echo_config(__file__, autoaudit_dir=AUTOAUDIT_DIR, results_dir=RESULTS_DIR, seeds=list(SEEDS))
    benchmark_synthetic()
