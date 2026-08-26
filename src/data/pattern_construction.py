import pandas as pd
import numpy as np

def create_identifiers(df):
    """
    Create a list of identifiers for each row in the dataframe.
    """
    # Convert all columns to string type
    df_str = df.astype(str)

    # Then use agg to join all column values into a single string for each row
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

        # Initialize all lists with zeros for simplification
        list_defaults = [0] * 8  # Assuming there are 8 lists as per the code snippet

        # Mapping of column names to their corresponding list index
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
            # Check if not at the end of the file
            if not line:
                break

            # Add pattern to the corresponding transaction
            if line.startswith("BEGIN"): # Start of a pattern
                attemptActive = True
                column = line.split(" - ")[1].split(":")[0].strip()
            elif line.startswith("END"): # End of a pattern => reset all parameters + no update of columns
                attemptActive = False
                column = ""
            elif attemptActive:
                identifyer = line.strip()
                transaction_list.append(identifyer)
                
                # Reset all lists to default values
                current_values = list_defaults.copy()
                
                if column in column_to_list_index:
                    # Update the relevant list based on the column name
                    current_values[column_to_list_index[column]] = 1
                    
                    # Unpack the updated values to each list
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

def define_ML_labels(path_trans="data/HI-Small_Trans.csv", path_patterns="data/HI-Small_Patterns.txt"):
    dtype_dict = {
            "From Bank": str,
            "To Bank": str,
            "Account": str,
            "Account.1": str
        }

    transactions_df = pd.read_csv(path_trans, dtype=dtype_dict)

    columns_money = ['Amount Received', 'Amount Paid']
    for col in columns_money: # make sure monetary amounts have two decimals
        transactions_df[col] = transactions_df[col].apply(lambda x: format_number(x))

    transactions_df['Is Laundering'] = transactions_df['Is Laundering'].astype(int)
    
    identifyer_list = create_identifiers(transactions_df)
    transactions_df["Identifyer"] = identifyer_list
    del identifyer_list

    pattern_columns = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"]
    df_patterns = create_AML_labels(path_patterns)

    # Merge the two dataframes
    transactions_df_extended = transactions_df.merge(df_patterns, on="Identifyer", how="left")
    transactions_df_extended = transactions_df_extended.fillna(0)

    # Vectorized check for "Is Laundering" being 1
    is_laundering = transactions_df_extended["Is Laundering"] == 1

    # Vectorized sum of specified columns
    pattern_sum = transactions_df_extended[pattern_columns].sum(axis=1)

    # Use numpy.where for a vectorized conditional operation
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
    # This function can take multiple columns as input and return a dictionary of scores for each column
    missing = [column for column in columns if column not in results_df.columns]
    if missing: # would otherwise come out as an all -2 feature, silently
        raise KeyError("columns absent from the GARG-AML results: "+str(missing))

    scores_dict = {column: [] for column in columns}

    for account in laundering_df.index:
        try: # one lookup per account, not one per column: task 3 asks for up to
             # 28 columns and this lookup is what the function spends its time on
            line = results_df.loc[account]
        except: # no GARG-AML row for this account
            for column in columns:
                scores_dict[column].append(-2)
            continue

        for column in columns:
            scores_dict[column].append(line[column])

    return scores_dict