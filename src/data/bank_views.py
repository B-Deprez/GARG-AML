"""
Single-bank views of the IBM data (task 5, partial observability).

Reviewers R1-1/R1-2 point out that no single institution observes the graph
GARG-AML scores: a bank sees only transactions with at least one of its own
customers on them, so a node's second-order neighbourhood -- the thing the
score is computed from -- may not exist from that bank's point of view. This
module builds those restricted views so the degradation can be measured
instead of conceded.

A view is a dataset
-------------------
The whole design rests on one observation about the existing code: every
output path in this repository is built from a ``dataset`` string
(``results/<dataset>_GARGAML_<direction>.csv``,
``results/<dataset>_folds.csv``, the metric matrices, ...). So a view is
given a *dataset name* of its own, ``<dataset>_bank<b>``, and it flows
through the pipeline as a plain string exactly like ``HI-Small`` does. The
scripts keep their existing "list of dataset names" configuration shape,
and ``src/utils/evaluation.py``, ``src/utils/features.py`` and
``src/utils/naming.py`` need no changes at all. :func:`parse_view` maps the
name back to the underlying data file plus the bank filter; ``bank=None``
reproduces today's behaviour byte for byte, so full-data results keep their
historical filenames.

Which transactions are in a view
--------------------------------
Filtering on the **bank fields** (``From Bank == b or To Bank == b``) and
filtering on **client membership** (``Account`` or ``Account.1`` belongs to
a client of *b*) select the same rows: verified on HI-Small for banks
``012``, ``070`` and ``001`` with zero differing rows. They can only diverge
on accounts that appear under two banks, of which HI-Small has 8 out of
515,080. The bank-field filter is used here because it is a vectorised mask
over two columns and needs no client set built first.

Who gets evaluated
------------------
The bank's **own clients**, not every account in the view -- see
:func:`bank_clients`. That is the population a bank actually alerts on, and
it removes a confound for free: every transaction involving client *c* has
*c* on one side, hence carries *c*'s bank in its bank field, hence is in the
view. So a client's laundering propensity computed on the view equals its
full-data value, the labels are untouched, and a view-vs-full comparison
varies only the features. External counterparties stay in the graph -- they
are what remains of the second-order neighbourhood -- they are simply not
scored.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from src.utils.graph_processing import strip_resolution

# The two columns a view is defined on, and the account columns they pair
# with. ``Account`` is booked at ``From Bank``, ``Account.1`` at ``To Bank``.
BANK_COLUMNS = ("From Bank", "To Bank")
ACCOUNT_COLUMNS = ("Account", "Account.1")

# Separator in the view's dataset name. Chosen to stay a legal filename and
# to be absent from every existing dataset name, so parse_view can invert
# view_name unambiguously.
VIEW_SEPARATOR = "_bank"

# Bank identifiers are zero-padded strings ("010" != "10"), so every read of
# a transactions file must force them to str -- pandas would otherwise infer
# int64 and silently drop the padding, making every bank lookup miss.
BANK_DTYPES = {c: str for c in BANK_COLUMNS + ACCOUNT_COLUMNS}


def normalise_banks(banks) -> list[str] | None:
    """Normalise a bank argument to a sorted list of strings, or ``None``.

    Accepts ``None`` (the full graph), a single identifier, or an iterable
    of them. Identifiers are coerced to ``str`` because they are zero-padded
    in the CSV and a caller passing ``12`` instead of ``"012"`` would
    otherwise select nothing at all rather than failing.
    """
    if banks is None:
        return None
    if isinstance(banks, (str, bytes)) or not isinstance(banks, Iterable):
        banks = [banks]
    out = sorted({str(b) for b in banks})
    if not out:
        return None
    return out


# A bank *group* spec stands for several banks at once. "top<k>" is the k
# banks with the most clients, which is how a realistically-sized
# institution is expressed: the largest single bank in HI-Small holds 0.512%
# of the accounts, where a pooled "top50" holds 10.8% -- closer to what a
# real large bank sees, and the difference decides whether the partial-
# observability experiment has enough positives to say anything.
GROUP_PREFIX = "top"

# Expanding a group costs a pass over the transactions file, and the graph,
# label and measure builders each resolve independently, so the result is
# cached per (path, spec).
_GROUP_CACHE: dict = {}


def is_bank_group(spec) -> bool:
    """True for a group spec such as ``"top50"``."""
    return (isinstance(spec, str) and spec.startswith(GROUP_PREFIX)
            and spec[len(GROUP_PREFIX):].isdigit())


def resolve_banks(banks, path=None):
    """Expand a bank spec into concrete identifiers.

    ``banks`` is either concrete -- one identifier or a list of them, in
    which case this is just :func:`normalise_banks` -- or a single group
    spec (see :data:`GROUP_PREFIX`), which needs ``path`` to expand.

    Call this once at the top of a script and pass the result down;
    :func:`bank_mask` expects concrete identifiers and would otherwise
    filter on the literal string "top50" and silently select nothing.
    """
    banks = normalise_banks(banks)
    if banks is None:
        return None
    if len(banks) == 1 and is_bank_group(banks[0]):
        spec = banks[0]
        if path is None:
            raise ValueError(
                "resolving the bank group " + repr(spec) + " needs the "
                "transactions file; pass path=..."
            )
        key = (path, spec)
        if key not in _GROUP_CACHE:
            k = int(spec[len(GROUP_PREFIX):])
            _GROUP_CACHE[key] = normalise_banks(client_counts(path).index[:k])
        return _GROUP_CACHE[key]
    return banks


def _require_resolved(banks, caller):
    """Normalise ``banks`` and reject a group spec that was never expanded.

    A group spec is a *name* for a set of banks, not a bank, so matching it
    against a bank column selects nothing. Every function below filters on
    such a column, and returning an empty result would be indistinguishable
    from the legitimate "this view has no positives" outcome the pipeline
    reports all the time -- a whole institution would silently evaluate zero
    accounts. Fail loudly instead; the caller owes us
    :func:`resolve_banks`, which needs the transactions file this frame no
    longer carries.
    """
    banks = normalise_banks(banks)
    if banks is None:
        return None
    unresolved = [b for b in banks if is_bank_group(b)]
    if unresolved:
        raise ValueError(
            caller + " needs concrete bank identifiers, but was given the "
            "unexpanded group spec(s) " + str(unresolved) + " -- call "
            "resolve_banks(banks, path) first and pass the result down"
        )
    return banks


def view_name(dataset: str, banks=None) -> str:
    """Dataset name for the view of ``dataset`` seen by ``banks``.

    ``banks=None`` returns ``dataset`` unchanged -- that is what keeps the
    full-data result files on their historical names.
    """
    banks = normalise_banks(banks)
    if banks is None:
        return dataset
    return dataset + VIEW_SEPARATOR + "-".join(banks)


def parse_view(name: str) -> tuple[str, list[str] | None]:
    """Inverse of :func:`view_name`: ``name`` -> ``(dataset, banks)``.

    Lets a script keep configuring itself with a flat list of dataset-name
    strings while still knowing which data file to read and which filter to
    apply.

    Task 4 decorates the same names with a Louvain setting
    (``HI-Small_res20``, ``HI-Small_nolouvain``), so the token is stripped
    here: this is the one place a dataset name is mapped back to its
    underlying data, and doing it here means :func:`trans_path`,
    :func:`patterns_path` and every ``base, banks = parse_view(...)`` caller
    keep working untouched. Use
    :func:`src.utils.graph_processing.parse_resolution` to read the setting
    itself.
    """
    if VIEW_SEPARATOR not in name:
        return strip_resolution(name), None
    dataset, _, banks = name.partition(VIEW_SEPARATOR)
    # The Louvain token may sit on either side of the bank separator, since
    # "HI-Small_res20_bank012" and "HI-Small_bank012_res20" are both natural
    # things to type; strip both parts rather than fixing an order.
    return strip_resolution(dataset), normalise_banks(
        strip_resolution(banks).split("-"))


def trans_path(name: str, directory: str = "data") -> str:
    """Transactions CSV backing ``name``, which may be a view name."""
    dataset, _ = parse_view(name)
    return directory + "/" + dataset + "_Trans.csv"


def patterns_path(name: str, directory: str = "data") -> str:
    """Patterns file backing ``name``, which may be a view name."""
    dataset, _ = parse_view(name)
    return directory + "/" + dataset + "_Patterns.txt"


def bank_mask(df: pd.DataFrame, banks) -> pd.Series:
    """Boolean mask of the transactions visible to ``banks``.

    ``True`` where the transaction was booked at one of ``banks`` on either
    side, i.e. where at least one endpoint is a client of one of them.
    """
    banks = _require_resolved(banks, "bank_mask")
    if banks is None:
        return pd.Series(True, index=df.index)

    missing = [c for c in BANK_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "bank columns absent from the transactions frame: " + str(missing)
            + " -- a bank view cannot be built from this file"
        )

    wanted = set(banks)
    mask = df[BANK_COLUMNS[0]].isin(wanted)
    for column in BANK_COLUMNS[1:]:
        mask |= df[column].isin(wanted)
    return mask


def filter_transactions(df: pd.DataFrame, banks) -> pd.DataFrame:
    """Restrict ``df`` to the view of ``banks``; identity when ``None``.

    Returns ``df`` itself (not a copy) for the full-graph case so the
    default path costs nothing.
    """
    if normalise_banks(banks) is None:
        return df
    return df[bank_mask(df, banks)]


def bank_clients(df: pd.DataFrame, banks) -> set:
    """Accounts held at ``banks`` -- the population a view is evaluated on.

    ``None`` returns every account in ``df``, so a caller can apply this
    unconditionally and get the full-data behaviour for free.
    """
    banks = _require_resolved(banks, "bank_clients")
    accounts = set()
    for bank_column, account_column in zip(BANK_COLUMNS, ACCOUNT_COLUMNS):
        if banks is None:
            accounts |= set(df[account_column])
        else:
            accounts |= set(df.loc[df[bank_column].isin(set(banks)), account_column])
    return accounts


def bank_account_pairs(path: str, chunksize: int | None = None) -> pd.DataFrame:
    """Distinct ``(bank, account)`` pairs in a transactions file.

    ``chunksize`` streams the file instead of loading it, which is what
    makes this usable on LI-Large's 16 GB CSV: each chunk is reduced to its
    distinct pairs before the next one is read, so peak memory is the chunk
    plus the running pair set rather than the file. HI-Small is small enough
    to leave ``chunksize=None``.
    """
    usecols = list(BANK_COLUMNS + ACCOUNT_COLUMNS)

    def pairs(frame):
        parts = [
            frame[[bank_column, account_column]].rename(
                columns={bank_column: "bank", account_column: "account"}
            )
            for bank_column, account_column in zip(BANK_COLUMNS, ACCOUNT_COLUMNS)
        ]
        return pd.concat(parts, ignore_index=True).drop_duplicates()

    if chunksize is None:
        return pairs(pd.read_csv(path, usecols=usecols, dtype=BANK_DTYPES))

    seen = []
    reader = pd.read_csv(path, usecols=usecols, dtype=BANK_DTYPES, chunksize=chunksize)
    for chunk in reader:
        seen.append(pairs(chunk))
    return pd.concat(seen, ignore_index=True).drop_duplicates()


def client_counts(path: str, chunksize: int | None = None) -> pd.Series:
    """Number of distinct clients per bank, descending.

    This is the quantity task 5's bank selection is made on. Note that it
    disagrees with transaction volume: on HI-Small, bank ``070`` has 15
    clients but 452,751 transactions and a larger view than the 2,639-client
    bank ``012``. Report both before choosing a "large" and a "small" bank.
    """
    pairs = bank_account_pairs(path, chunksize=chunksize)
    return pairs.groupby("bank")["account"].nunique().sort_values(ascending=False)
