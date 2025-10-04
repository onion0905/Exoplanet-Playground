#!/usr/bin/env python3
"""Quick script to check dataset columns"""

import sys
from pathlib import Path
import pandas as pd

# Add ML directory to Python path
ml_dir = Path(__file__).parent / "ML"
sys.path.insert(0, str(ml_dir))

from src.data.data_loader import DataLoader

loader = DataLoader()

for dataset in ['kepler', 'tess', 'k2']:
    print(f"\n=== {dataset.upper()} DATASET ===")
    try:
        df = loader.load_nasa_dataset(dataset)
        print(f"Shape: {df.shape}")
        print("Columns:")
        for col in df.columns:
            if 'disposition' in col.lower() or 'status' in col.lower():
                print(f"  🎯 {col} (potential target)")
            else:
                print(f"     {col}")
        
        # Check for common target columns
        target_candidates = [col for col in df.columns if 'disposition' in col.lower() or 'status' in col.lower()]
        if target_candidates:
            print(f"Target candidates: {target_candidates}")
            # Show unique values
            for target in target_candidates[:2]:  # Just first 2
                print(f"  {target} values: {df[target].value_counts().index.tolist()[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")