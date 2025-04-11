import json
import os


def get_min_max(series):
    return series.min(), series.max()

def custom_min_max_normalization(data, X_min, X_max, min_new=1, max_new=3):
    return [(x - X_min) / (X_max - X_min) * (max_new - min_new) + min_new for x in data]

def save_min_max_values(min_max_values, file_name):
    with open(file_name, 'w') as f:
        json.dump(min_max_values, f)

def read_min_max_values(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def normalize_continuous_variables(datasets, variable_names, min_new=1, max_new=3, save_path=None, read_path=None):
    """
    Normalize continuous variables using min-max scaling.
    Either compute and save min-max values (using save_path) or read existing ones (using read_path).

    Args:
        datasets (List[pd.DataFrame]): List of datasets to normalize
        variable_names (List[str]): List of numerical variable names
        min_new (float): New minimum for normalization range
        max_new (float): New maximum for normalization range
        save_path (str): Path to save min-max JSON file
        read_path (str): Path to load min-max JSON file

    Returns:
        List[pd.DataFrame]: Normalized datasets
    """
    if (save_path and read_path) or (not save_path and not read_path):
        raise ValueError("Provide exactly one of save_path or read_path.")

    if save_path:
        min_max_values = {var: get_min_max(datasets[0][var]) for var in variable_names}
        save_min_max_values(min_max_values, save_path)
        print(f"[INFO] Saved min/max values to {save_path}")
    elif read_path:
        min_max_values = read_min_max_values(read_path)
        print(f"[INFO] Loaded min/max values from {read_path}")

    for dataset in datasets:
        for var in variable_names:
            X_min, X_max = min_max_values[var]
            dataset[var] = custom_min_max_normalization(dataset[var], X_min, X_max, min_new, max_new)

    return datasets

