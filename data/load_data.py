import os
import pandas as pd
from config.paths import DATA_PATH, DEVEL_CSV, TEST_CSV

def load_datasets(shuffle: bool = True, train_frac: float = 0.8, seed: int = 1234):
    # Load devel/test CSV
    devel_df = pd.read_csv(os.path.join(DATA_PATH, DEVEL_CSV))
    test_df = pd.read_csv(os.path.join(DATA_PATH, TEST_CSV))

    # Extract variable names
    numerical_variable_names = [col for col in devel_df.columns if col.endswith("_Continuous") or col.endswith("_Integer")]
    categorical_variable_names = [col for col in devel_df.columns if col.endswith("_Categorical")]

    # Subset necessary columns
    use_columns = ["f.eid", "CVD"] + numerical_variable_names + categorical_variable_names
    devel_df = devel_df[use_columns]
    test_df = test_df[use_columns]

    # Split devel into train/val
    if shuffle:
        devel_df = devel_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    split = int(len(devel_df) * train_frac)
    train_df = devel_df[:split].reset_index(drop=True)
    val_df = devel_df[split:].reset_index(drop=True)

    return train_df, val_df, test_df, numerical_variable_names, categorical_variable_names

