#!/usr/bin/env python3
"""Quick check for TESS target column"""

import sys
from pathlib import Path
ml_dir = Path(__file__).parent / "ML"
sys.path.insert(0, str(ml_dir))

from src.data.data_loader import DataLoader
loader = DataLoader()
df = loader.load_nasa_dataset('tess')

print("TESS columns with disposition/status:")
for col in df.columns:
    if 'disp' in col.lower() or 'status' in col.lower():
        print(f"  {col}: {df[col].nunique()} unique values")
        print(f"    Values: {df[col].value_counts().index.tolist()[:5]}")
        print(f"    Non-null count: {df[col].notna().sum()}/{len(df)}")
        print()