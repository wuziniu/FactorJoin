import pandas as pd
import pickle
from Join_scheme.data_prepare import process_stats_data
from BayesCard.Models.Bayescard_BN import Bayescard_BN

def train_DMV(csv_path, model_path, algorithm, max_parents, sample_size):
    data = pd.read_csv(csv_path)
    new_cols = []
    #removing unuseful columns
    for col in data.columns:
        if col in ['VIN', 'Zip', 'City', 'Make', 'Unladen Weight', 'Maximum Gross Weight', 'Passengers',
                   'Reg Valid Date', 'Reg Expiration Date', 'Color']:
            data = data.drop(col, axis=1)
        else:
            new_cols.append(col.replace(" ", "_"))
    data.columns = new_cols
    BN = Bayescard_BN('dmv')
    BN.build_from_data(data, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'], sample_size=sample_size)
    model_path += f"/{algorithm}_{max_parents}.pkl"
    pickle.dump(BN, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"model saved at {model_path}")
    return None

def train_Census(csv_path, model_path, algorithm, max_parents, sample_size):
    df = pd.read_csv(csv_path, header=0, sep=",")
    df = df.drop("caseid", axis=1)
    df = df.dropna(axis=0)
    BN = Bayescard_BN('Census')
    BN.build_from_data(df, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'], sample_size=sample_size)
    model_path += f"/{algorithm}_{max_parents}.pkl"
    pickle.dump(BN, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"model saved at {model_path}")
    return None


def train_stats(data_path, model_folder, n_bins=500, save_bucket_bins=False):
    data, null_values, key_attrs, all_bin_modes = process_stats_data(data_path, model_folder, n_bins, save_bucket_bins)
    for table in data:
        print(f"training BayesCard on table {table}")
        bn = Bayescard_BN(table, key_attrs[table], null_values=null_values[table])
        bn.build_from_data(data[table])
        model_path = model_folder + f"/{table}.pkl"
        pickle.dump(bn, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"model saved at {model_path}")

