"""
Single-bank views of the IBM data (partial observability).

A bank observes only the transactions that have one of its own clients on
them, so an account's second-order neighbourhood -- what the GARG-AML score
is computed from -- is partly invisible from that bank's point of view. This
module builds those restricted views.

A view is a dataset
-------------------
A view is given a *dataset name* of its own, ``<dataset>_bank<b>``, and
flows through the pipeline as a plain string exactly like ``HI-Small``
does. :func:`parse_view` maps the name back to the underlying data file plus
the bank filter; ``banks=None`` is the full data.

Which transactions are in a view
--------------------------------
Filtering on the **bank fields** (``From Bank == b or To Bank == b``) and
filtering on **client membership** (``Account``/``Account.1`` belongs to a
client of *b*) select the same rows; they diverge only on an account held at
two banks. The bank-field filter is used since it's a vectorised mask over
two columns, needing no client set built first.

Who gets evaluated
------------------
The bank's **own clients**, not every account in the view -- see
:func:`bank_clients`. Every transaction involving client *c* has *c* on one
side, hence carries *c*'s bank in its bank field, hence is in the view. So a
client's laundering propensity computed on the view equals its full-data
value: the labels are identical and a view-vs-full comparison varies only
the features. External counterparties stay in the graph -- what remains of
the second-order neighbourhood -- but are not scored.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from src.utils.graph_processing import strip_resolution

# Columns a view is defined on, paired with the account column each books to:
# ``Account`` at ``From Bank``, ``Account.1`` at ``To Bank``.
BANK_COLUMNS = ("From Bank", "To Bank")
ACCOUNT_COLUMNS = ("Account", "Account.1")

# Legal filename character, absent from every dataset name, so parse_view
# inverts view_name unambiguously.
VIEW_SEPARATOR = "_bank"

# Bank identifiers are zero-padded strings ("010" != "10"); force str here or
# pandas infers int64, drops the padding, and every bank lookup misses.
BANK_DTYPES = {c: str for c in BANK_COLUMNS + ACCOUNT_COLUMNS}


def normalise_banks(banks) -> list[str] | None:
    """Normalise a bank argument to a sorted list of strings, or ``None``.

    Accepts ``None`` (full graph), a single identifier, or an iterable of
    them. Identifiers are coerced to ``str`` since they're zero-padded in the
    CSV -- passing ``12`` instead of ``"012"`` would otherwise silently
    select nothing rather than fail.
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
# banks with the most clients, pooled as one institution: no individual bank
# in the data holds a realistic share of the accounts.
GROUP_PREFIX = "top"

# Expanding a group costs a pass over the transactions file, and the graph,
# label and measure builders each resolve independently, so cache by (path, spec).
_GROUP_CACHE: dict = {}


def is_bank_group(spec) -> bool:
    """True for a group spec such as ``"top50"``."""
    return (isinstance(spec, str) and spec.startswith(GROUP_PREFIX)
            and spec[len(GROUP_PREFIX):].isdigit())


def resolve_banks(banks, path=None):
    """Expand a bank spec into concrete identifiers.

    ``banks`` is either concrete -- one identifier or a list, in which case
    this is just :func:`normalise_banks` -- or a single group spec (see
    :data:`GROUP_PREFIX`), which needs ``path`` to expand. Every other
    function here expects the concrete identifiers this returns.
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
    against a bank column selects nothing -- indistinguishable from a view
    that legitimately has no positives. Callers must expand the spec with
    :func:`resolve_banks` first, since this function has no transactions
    file to do it with.
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

    ``banks=None`` returns ``dataset`` unchanged, so full-data results keep
    the plain dataset name.
    """
    banks = normalise_banks(banks)
    if banks is None:
        return dataset
    return dataset + VIEW_SEPARATOR + "-".join(banks)


def parse_view(name: str) -> tuple[str, list[str] | None]:
    """Inverse of :func:`view_name`: ``name`` -> ``(dataset, banks)``.

    Lets a script configure itself with a flat list of dataset-name strings
    while still knowing which data file to read and which filter to apply.

    A name may also carry a Louvain setting (``HI-Small_res20``,
    ``HI-Small_nolouvain``) on either side of the bank separator; that token
    is stripped here. Use
    :func:`src.utils.graph_processing.parse_resolution` to read the setting
    itself.
    """
    if VIEW_SEPARATOR not in name:
        return strip_resolution(name), None
    dataset, _, banks = name.partition(VIEW_SEPARATOR)
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

    Returns ``df`` itself, not a copy, for the full-graph case.
    """
    if normalise_banks(banks) is None:
        return df
    return df[bank_mask(df, banks)]


def bank_clients(df: pd.DataFrame, banks) -> set:
    """Accounts held at ``banks`` -- the population a view is evaluated on.

    ``None`` returns every account in ``df``.
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

    ``chunksize`` streams the file instead of loading it whole -- what makes
    this usable on the multi-gigabyte LI-Large CSV: each chunk is reduced to
    its distinct pairs before the next is read, so peak memory is one chunk
    plus the running pair set. ``None`` loads the file whole.
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

    This is the quantity banks are selected on. It does not order banks the
    same way transaction volume does: a bank with few clients can still
    carry many transactions, and hence a large view.
    """
    pairs = bank_account_pairs(path, chunksize=chunksize)
    return pairs.groupby("bank")["account"].nunique().sort_values(ascending=False)
