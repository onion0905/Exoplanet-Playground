#!/usr/bin/env python3
"""
Test API with high-performance settings to match train_rf_clean.py results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ML', 'src'))

import pandas as pd
from api.training_api import TrainingAPI

def test_high_performance_training():
    """Test the API with settings that match train_rf_clean.py high performance."""
    
    # Initialize API
    api = TrainingAPI()
    
    # Load the same raw data that train_rf_clean.py uses
    print("Loading raw Kepler data...")
    df = pd.read_csv('data/kepler_raw.csv', comment='#')
    print(f"Loaded data shape: {df.shape}")
    
    # Apply NASA API filtering but with minimal missing value filtering
    print("\nApplying NASA API filtering...")
    df_filtered, excluded_cols = api.data_processor.apply_nasa_api_filtering(df)
    print(f"After NASA filtering: {df_filtered.shape}")
    print(f"Excluded columns ({len(excluded_cols)}): {excluded_cols}")
    
    # Configure training with high-performance parameters matching train_rf_clean.py
    print("\nConfiguring high-performance training...")
    config = {
        'target_column': 'koi_disposition',
        'model_type': 'random_forest',
        'use_nasa_filtering': False,  # Already filtered above
        'preprocessing_config': {
            'max_missing_ratio': 0.999,  # Use almost all features like train_rf_clean.py
            'handle_missing': True,
            'missing_strategy': 'median',  # Same as train_rf_clean.py
            'scale_features': False,  # train_rf_clean.py doesn't scale by default
            'encode_categorical': True,
            'remove_outliers': False
        },
        'hyperparameters': {
            'n_estimators': 300,  # Same as high-performance train_rf_clean.py
            'max_depth': 20,      # Same as high-performance train_rf_clean.py
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,   # Same random state
            'n_jobs': -1
        }
    }
    
    # Get all available feature columns (like train_rf_clean.py does)
    target_col = 'koi_disposition'
    all_features = [col for col in df_filtered.columns if col != target_col]
    config['feature_columns'] = all_features
    
    print(f"Using {len(all_features)} features (matching train_rf_clean.py approach)")
    
    session_id = api.quick_configure_training(df_filtered, config)
    print(f"Training configured. Session ID: {session_id}")
    
    # Train with high-performance model config
    model_config = {
        'model_type': 'random_forest',
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"\nTraining high-performance model...")
    training_results = api.quick_train_model(session_id, model_config)
    
    # Extract results
    if training_results.get('status') == 'success':
        test_accuracy = training_results.get('evaluation_metrics', {}).get('accuracy')
        validation_accuracy = training_results.get('validation_accuracy')
        
        print(f"\n{'='*60}")
        print(f"HIGH-PERFORMANCE API RESULTS")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        if validation_accuracy:
            print(f"Validation Accuracy: {validation_accuracy:.4f}")
        
        # Compare with train_rf_clean.py benchmark
        print(f"\n{'='*60}")
        print(f"COMPARISON WITH train_rf_clean.py")
        print(f"{'='*60}")
        print(f"train_rf_clean.py accuracy: 0.9460 (benchmark)")
        print(f"API accuracy:               {test_accuracy:.4f}")
        print(f"Difference:                 {test_accuracy - 0.9460:+.4f}")
        
        if abs(test_accuracy - 0.9460) < 0.01:
            print("✅ SUCCESS: API matches train_rf_clean.py performance!")
        else:
            print("❌ ISSUE: API performance differs significantly from benchmark")
            
        # Get session info for detailed analysis
        session_info = api.get_session_info(session_id, include_data=True)
        if session_info['status'] == 'success':
            session_data = session_info['session_info']
            if 'prepared_data' in session_data:
                prepared = session_data['prepared_data']
                print(f"\nFeature analysis:")
                print(f"  Features used in training: {prepared['X_train'].shape[1]}")
                print(f"  Training samples: {len(prepared['X_train'])}")
                print(f"  Test samples: {len(prepared['X_test'])}")
                if 'X_val' in prepared and prepared['X_val'] is not None:
                    print(f"  Validation samples: {len(prepared['X_val'])}")
                
    else:
        print(f"❌ Training failed: {training_results.get('error', 'Unknown error')}")
        
    return training_results

if __name__ == "__main__":
    test_high_performance_training()