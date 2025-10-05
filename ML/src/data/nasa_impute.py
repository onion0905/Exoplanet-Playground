import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

NASA_DATASET_PATHS = {
    'kepler': Path(__file__).parent.parent.parent / 'data' / 'kepler_raw.csv',
    'k2': Path(__file__).parent.parent.parent / 'data' / 'k2_raw.csv',
    'tess': Path(__file__).parent.parent.parent / 'data' / 'tess_raw.csv',
}

def get_column_means(dataset_type: str) -> Dict[str, float]:
    """
    Load the original NASA dataset and compute column means for numeric columns.
    Returns a dict mapping column name to mean value.
    """
    dataset_type = dataset_type.lower()
    if dataset_type not in NASA_DATASET_PATHS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    path = NASA_DATASET_PATHS[dataset_type]
    df = pd.read_csv(path, comment='#')
    means = df.mean(numeric_only=True).to_dict()
    return means

def impute_with_nasa_means(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Fill missing values in df with column means from the original NASA dataset.
    Only applies to columns present in both df and the NASA dataset.
    """
    means = get_column_means(dataset_type)
    for col in df.columns:
        if col in means:
            df[col] = df[col].fillna(means[col])
    return df
