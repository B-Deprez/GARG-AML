import pandas as pd
import numpy as np

from src.data.bank_views import filter_transactions, resolve_banks

def create_identifiers(df):
    """
    Create a list of identifiers for each row in the dataframe.

    The identifier is the comma-joined row, which is the form a transaction
    takes inside a ``*_Patterns.txt`` file.
    """
    df_str = df.astype(str)

    identifyer_list = df_str.agg(','.join, axis=1).tolist()

    return identifyer_list

def format_number(number):
    formatted = str(number)
    if not('.' in formatted and len(formatted.split('.')[1]) >= 2):
        formatted = format(number, '.2f')
    return formatted

def create_AML_labels(path= "data/HI-Small_Patterns.txt"):
    transaction_list = []
    fanout_list = []
    fanin_list = []
    gather_scatter_list = []
    scatter_gather_list = []
    cycle_list = []
    random_list = []
    bipartite_list = []
    stack_list = []

    with open(path, "r") as f:
        attemptActive = False
        column = ""

        list_defaults = [0] * 8

        column_to_list_index = {
            "FAN-OUT": 0,
            "FAN-IN": 1,
            "GATHER-SCATTER": 2,
            "SCATTER-GATHER": 3,
            "CYCLE": 4,
            "RANDOM": 5,
            "BIPARTITE": 6,
            "STACK": 7
        }
        while True:
            line = f.readline()
            if not line:
                break

            # A pattern block runs from its BEGIN line to its END line; the
            # lines in between are the transactions carrying that pattern.
            if line.startswith("BEGIN"):
                attemptActive = True
                column = line.split(" - ")[1].split(":")[0].strip()
            elif line.startswith("END"):
                attemptActive = False
                column = ""
            elif attemptActive:
                identifyer = line.strip()
                transaction_list.append(identifyer)

                current_values = list_defaults.copy()

                if column in column_to_list_index:
                    current_values[column_to_list_index[column]] = 1

                    fanout_list.append(current_values[0])
                    fanin_list.append(current_values[1])
                    gather_scatter_list.append(current_values[2])
                    scatter_gather_list.append(current_values[3])
                    cycle_list.append(current_values[4])
                    random_list.append(current_values[5])
                    bipartite_list.append(current_values[6])
                    stack_list.append(current_values[7])

                else:
                    raise ValueError("Unknown pattern type")
                
    df_patterns = pd.DataFrame(
        {
            "Identifyer": transaction_list,
            "FAN-OUT": fanout_list,
            "FAN-IN": fanin_list,
            "GATHER-SCATTER": gather_scatter_list,
            "SCATTER-GATHER": scatter_gather_list,
            "CYCLE": cycle_list,
            "RANDOM": random_list,
            "BIPARTITE": bipartite_list,
            "STACK": stack_list
        }
    )

    return df_patterns

# Column positions in a *_Patterns.txt transaction line -- verbatim rows of
# the matching *_Trans.csv (Timestamp, From Bank, Account, To Bank,
# Account.1, Amount Received, Receiving Currency, Amount Paid, Payment
# Currency, Payment Format, Is Laundering).
PATTERN_FROM_ACCOUNT = 2
PATTERN_TO_ACCOUNT = 4


def pattern_instances(path="data/HI-Small_Patterns.txt"):
    """One row per transaction, tagged with the laundering attempt it belongs to.

    :func:`create_AML_labels` reads the same file but flattens it, keeping
    only which pattern *type* each transaction carries. This parser keeps the
    ``BEGIN``/``END`` boundary as an ``instance`` counter, so the accounts of
    a single attempt -- each a connected money-flow structure of its own --
    can be recovered.

    Returns a DataFrame with ``instance``, ``pattern_type``, ``source`` and
    ``target``; the two GARG-AML targets (GATHER-SCATTER, SCATTER-GATHER)
    are selected on ``pattern_type``.

    Accounts are returned as the raw strings in the file, which is what
    ``construct_IBM_graph`` uses for node identity -- it reads the account
    columns without forcing a dtype, and they contain hexadecimal-looking
    values, so pandas infers ``object``.
    """
    rows = []
    instance = -1
    pattern_type = ""

    with open(path, "r") as handle:
        for line in handle:
            if line.startswith("BEGIN"):
                instance += 1
                pattern_type = line.split(" - ")[1].split(":")[0].strip()
            elif line.startswith("END"):
                pattern_type = ""
            elif pattern_type:
                fields = line.strip().split(",")
                if len(fields) <= PATTERN_TO_ACCOUNT:
                    continue  # blank/malformed line inside an attempt
                rows.append({
                    "instance": instance,
                    "pattern_type": pattern_type,
                    "source": fields[PATTERN_FROM_ACCOUNT],
                    "target": fields[PATTERN_TO_ACCOUNT],
                })

    return pd.DataFrame(rows, columns=["instance", "pattern_type",
                                       "source", "target"])


def define_ML_labels(path_trans="data/HI-Small_Trans.csv", path_patterns="data/HI-Small_Patterns.txt", banks=None):
    """Per-transaction laundering labels, optionally under a single-bank view.

    ``banks`` keeps only the transactions booked at those banks. The
    patterns file is read whole either way and joined on the transaction
    identifier, so a view simply matches fewer of its rows -- there is no
    separate patterns file per view. ``None`` is the full data.
    """
    dtype_dict = {
            "From Bank": str,
            "To Bank": str,
            "Account": str,
            "Account.1": str
        }

    banks = resolve_banks(banks, path_trans)  # expands a group spec such as "top50"

    transactions_df = pd.read_csv(path_trans, dtype=dtype_dict)
    # Filter before the identifier/pattern join: the identifier is built from
    # the row's own columns, so dropping rows first is equivalent and keeps
    # the join off transactions a view never sees.
    transactions_df = filter_transactions(transactions_df, banks)

    columns_money = ['Amount Received', 'Amount Paid']
    for col in columns_money:  # enforce two decimals on monetary amounts
        transactions_df[col] = transactions_df[col].apply(lambda x: format_number(x))

    transactions_df['Is Laundering'] = transactions_df['Is Laundering'].astype(int)
    
    identifyer_list = create_identifiers(transactions_df)
    transactions_df["Identifyer"] = identifyer_list
    del identifyer_list

    pattern_columns = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"]
    df_patterns = create_AML_labels(path_patterns)

    transactions_df_extended = transactions_df.merge(df_patterns, on="Identifyer", how="left")
    transactions_df_extended = transactions_df_extended.fillna(0)

    is_laundering = transactions_df_extended["Is Laundering"] == 1

    pattern_sum = transactions_df_extended[pattern_columns].sum(axis=1)

    # A laundering transaction that matches no pattern block in the patterns
    # file gets its own bucket.
    transactions_df_extended["Not Classified"] = np.where((is_laundering) & (pattern_sum == 0), 1, 0)
        
    pattern_columns.append("Not Classified")

    return transactions_df_extended, pattern_columns

def summarise_ML_labels(transactions_df_extended, pattern_columns):
    laundering_from = transactions_df_extended[["Account", "Is Laundering"]+pattern_columns].groupby("Account").mean()
    laundering_to = transactions_df_extended[["Account.1", "Is Laundering"]+pattern_columns].groupby("Account.1").mean()
    
    trans_from=transactions_df_extended[["Account", "Is Laundering"]+pattern_columns]
    trans_to=transactions_df_extended[["Account.1", "Is Laundering"]+pattern_columns]
    trans_to.columns = ["Account", "Is Laundering"]+pattern_columns
    laundering_combined = pd.concat([trans_from, trans_to]).groupby("Account").mean()

    return laundering_combined, laundering_from, laundering_to

def combine_patterns_GARGAML(results_df, laundering_df, columns = ["GARGAML"]):
    # Accounts without a GARG-AML row get -2, outside the score's [-1, 1]
    # range.
    missing = [column for column in columns if column not in results_df.columns]
    if missing:  # would otherwise silently come out as an all -2 feature
        raise KeyError("columns absent from the GARG-AML results: "+str(missing))

    scores_dict = {column: [] for column in columns}

    for account in laundering_df.index:
        try:  # one lookup per account, not per column -- that's the cost
            line = results_df.loc[account]
        except:  # no GARG-AML row for this account
            for column in columns:
                scores_dict[column].append(-2)
            continue

        for column in columns:
            scores_dict[column].append(line[column])

    return scores_dict