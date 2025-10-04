#!/usr/bin/env python3
"""Debug script to test TrainingAPI paths"""

import sys
from pathlib import Path

# Add ML directory to Python path
ml_dir = Path(__file__).parent / "ML"
sys.path.insert(0, str(ml_dir))

from src.api.training_api import TrainingAPI
from src.config import DATA_DIR, MODELS_DIR, MODEL_SAVE_DIR

print("=== TRAINING API PATH DEBUG ===")
print(f"CONFIG DATA_DIR: {DATA_DIR}")
print(f"CONFIG MODELS_DIR: {MODELS_DIR}")
print(f"CONFIG MODEL_SAVE_DIR: {MODEL_SAVE_DIR}")

print(f"\n=== TRAINING API TEST ===")
try:
    training_api = TrainingAPI()
    print(f"✅ TrainingAPI initialized successfully")
    print(f"TrainingAPI data_loader.data_dir: {training_api.data_loader.data_dir}")
    print(f"Data directory exists: {training_api.data_loader.data_dir.exists()}")
    
    # Test session creation
    session_result = training_api.start_training_session("test_session")
    print(f"✅ Session created: {session_result}")
    
    # Test data loading
    try:
        load_result = training_api.load_data_for_training(
            "test_session", 
            "nasa", 
            {"datasets": ["kepler"]}
        )
        print(f"✅ Data loaded successfully")
        print(f"Load result keys: {list(load_result.keys())}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        
except Exception as e:
    print(f"❌ TrainingAPI failed: {e}")
    import traceback
    traceback.print_exc()