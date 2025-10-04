#!/usr/bin/env python3
"""Debug script to check data paths"""

import sys
from pathlib import Path

# Add ML directory to Python path
ml_dir = Path(__file__).parent / "ML"
sys.path.insert(0, str(ml_dir))

from src.config import DATA_DIR, MODELS_DIR, NASA_DATASETS
from src.data.data_loader import DataLoader

print("=== PATH DEBUG ===")
print(f"DATA_DIR from config: {DATA_DIR}")
print(f"DATA_DIR exists: {DATA_DIR.exists()}")
print(f"MODELS_DIR: {MODELS_DIR}")

print(f"\n=== EXPECTED FILES ===")
for name, filename in NASA_DATASETS.items():
    filepath = DATA_DIR / filename
    print(f"{name}: {filepath} (exists: {filepath.exists()})")

print(f"\n=== DATA_LOADER TEST ===")
loader = DataLoader()
print(f"DataLoader data_dir: {loader.data_dir}")
print(f"DataLoader data_dir exists: {loader.data_dir.exists()}")

if loader.data_dir.exists():
    print("Files in DataLoader data_dir:")
    for file in loader.data_dir.iterdir():
        if file.is_file():
            print(f"  - {file.name}")

# Try to load one dataset
try:
    print(f"\n=== LOADING TEST ===")
    df = loader.load_nasa_dataset('kepler')
    print(f"✅ Successfully loaded kepler dataset: {df.shape}")
except Exception as e:
    print(f"❌ Failed to load kepler dataset: {e}")