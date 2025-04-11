import pandas as pd
from typing import List

def load_missing_pattern(file_path: str, top_n: int = 20) -> List[List[str]]:
    """
    Load missing variable patterns from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing missing variable information.
        top_n (int): Number of top missing patterns to return.

    Returns:
        List[List[str]]: A list of variable name lists to use for masking.
    """
    missing_info = pd.read_csv(file_path)
    missing_info = missing_info.iloc[:top_n]
    
    patterns = []
    for i in range(len(missing_info)):
        row = missing_info.iloc[i, 0]
        pattern = row.split(" ")
        patterns.append(pattern)
    
    return patterns

