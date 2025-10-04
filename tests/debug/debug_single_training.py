#!/usr/bin/env python3
"""Simple test to debug training workflow issues"""

import sys
from pathlib import Path
import traceback

# Add ML directory to Python path
ml_dir = Path(__file__).parent / "ML"
sys.path.insert(0, str(ml_dir))

from src.api.training_api import TrainingAPI

print("=== SINGLE MODEL TRAINING TEST ===")

training_api = TrainingAPI()
session_id = "debug_session"

try:
    print("1. Starting session...")
    session_result = training_api.start_training_session(session_id)
    print(f"✅ Session: {session_result}")
    
    print("2. Loading data...")
    load_result = training_api.load_data_for_training(
        session_id, 
        "nasa", 
        {"datasets": ["kepler"]}
    )
    print(f"✅ Data loaded: {list(load_result.keys())}")
    
    print("3. Configuring training...")
    config_result = training_api.configure_training(session_id, {
        'model_type': 'random_forest',
        'target_column': 'koi_disposition',
        'hyperparameters': {
            'n_estimators': 5,  # Very small for testing
            'max_depth': 3
        }
    })
    print(f"✅ Training configured: {list(config_result.keys())}")
    
    print("4. Starting training...")
    training_result = training_api.start_training(session_id)
    print(f"Training result: {training_result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Full traceback:")
    traceback.print_exc()