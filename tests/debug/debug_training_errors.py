#!/usr/bin/env python3

import sys
import os
import traceback
from ML.src.api.training_api import TrainingAPI
from ML.src.config import BASE_DIR

def debug_training_step_by_step():
    """Debug each step of the training process individually"""
    
    print("=== Debugging Training Process ===")
    
    try:
        # Test 1: Initialize TrainingAPI
        print("\n1. Initializing TrainingAPI...")
        api = TrainingAPI()
        print("✓ TrainingAPI initialized successfully")
        
        # Test 2: Start training session
        print("\n2. Starting training session...")
        session_id = f"debug_session_{int(__import__('time').time())}"
        result = api.start_training_session(session_id)
        print(f"Session ID: {session_id}")
        print(f"Result: {result}")
        
        # Test 3: Load data
        print("\n3. Loading data...")
        result = api.load_data_for_training(session_id, "kepler")
        print(f"Data loading result: {result}")
        
        # Test 4: Configure training
        print("\n4. Configuring training...")
        config = {
            'model_type': 'decision_tree',
            'target_column': 'koi_disposition',
            'feature_columns': ['koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_dror', 'koi_count', 'koi_num_transits'],
            'test_size': 0.2,
            'random_state': 42
        }
        
        result = api.configure_training(session_id, config)
        print(f"Configuration result: {result}")
        
        # Test 5: Start training (this is where errors likely occur)
        print("\n5. Starting model training...")
        result = api.start_training(session_id)
        print(f"Training result: {result}")
        
        if result['success']:
            print("✓ Training completed successfully!")
            print(f"Model ID: {result['model_id']}")
        else:
            print(f"✗ Training failed: {result}")
            
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_step_by_step()