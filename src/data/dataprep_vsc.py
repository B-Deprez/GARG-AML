# LOAD MODULES
# Standard library
import pandas as pd

def main(split=True):
    k=30

    dtype_dict = {
        "From Bank": str,
        "To Bank": str,
        "Account": str,
        "Account.1": str
    }

    if split:
        full_data_set = pd.read_csv("data/LI-Large_Trans.csv", dtype = dtype_dict)

        n = int(len(full_data_set)/k) # We devide the dataset in k parts
        for i in range(k):
            if i == k-1:
                full_data_set.iloc[i*n:].to_csv("data/LI-Large_Trans_"+str(i)+".csv", index=False)
            else:
                full_data_set.iloc[i*n:(i+1)*n].to_csv("data/LI-Large_Trans_"+str(i)+".csv", index=False)
            
    else:
        full_data_set = pd.DataFrame()
        columns_money = ['Amount Received', 'Amount Paid']
        
        for i in range(k):
            df_piece = pd.read_csv("data/LI-Large_Trans_"+str(i)+".csv", dtype = dtype_dict)

            for col in columns_money: # make sure monetary amounts have two decimals
                df_piece[col] = df_piece[col].astype(float).apply(lambda x: format(x, '.2f')).astype(str)

            full_data_set = pd.concat([full_data_set, df_piece])
        full_data_set.reset_index(drop=True, inplace=True)

        full_data_set.to_csv("data/LI-Large_Trans_vsc.csv", index=False)

main()
