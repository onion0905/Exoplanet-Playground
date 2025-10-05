#!/usr/bin/env python3
"""
Test API with RAW data to match train_rf_clean.py performance exactly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ML', 'src'))

import pandas as pd
from api.training_api import TrainingAPI

def test_raw_data_performance():
    """Test the API with raw data that train_rf_clean.py uses."""
    
    datasets = {
        'kepler': 'kepler_raw.csv',
        'k2': 'k2_raw.csv', 
        'tess': 'tess_raw.csv'
    }
    
    results = {}
    
    for dataset_name, filename in datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing {dataset_name.upper()} Dataset with RAW data")
        print(f"{'='*60}")
        
        try:
            # Initialize API
            api = TrainingAPI()
            
            # Load RAW data (same as train_rf_clean.py)
            print(f"Loading raw data: {filename}")
            df = pd.read_csv(f'data/{filename}', comment='#')
            print(f"Raw data shape: {df.shape}")
            
            # Apply NASA API filtering
            df_filtered, excluded_cols = api.data_processor.apply_nasa_api_filtering(df)
            print(f"After NASA filtering: {df_filtered.shape}")
            print(f"Excluded columns ({len(excluded_cols)}): {len(excluded_cols)} total")
            
            # Get all available feature columns (like train_rf_clean.py)
            detected_type, target_col = api.data_processor.column_filter.detect_dataset_type(df_filtered)
            all_features = [col for col in df_filtered.columns if col != target_col]
            print(f"Using {len(all_features)} features")
            
            # Check target distribution
            if target_col in df_filtered.columns:
                target_series = df_filtered[target_col]
                target_mapped = api.data_processor.column_filter.map_target_labels(target_series, detected_type)
                print(f"Target distribution:")
                print(target_mapped.value_counts())
            
            # Configure training with HIGH-PERFORMANCE settings
            config = {
                'target_column': target_col,
                'feature_columns': all_features,
                'model_type': 'random_forest',
                'use_nasa_filtering': False,  # Already filtered above
                'preprocessing_config': {
                    'max_missing_ratio': 0.999,  # Only exclude 100% missing
                    'handle_missing': True,
                    'missing_strategy': 'median',  # Same as train_rf_clean.py
                    'scale_features': False,       # train_rf_clean.py doesn't scale
                    'encode_categorical': True,
                    'remove_outliers': False
                }
            }
            
            session_id = api.quick_configure_training(df_filtered, config)
            print(f"Training configured. Session ID: {session_id}")
            
            # Train with HIGH-PERFORMANCE model config (same as train_rf_clean.py benchmarks)
            model_config = {
                'model_type': 'random_forest',
                'n_estimators': 300,  # High-performance setting
                'max_depth': 20,      # High-performance setting
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,   # Same random state
                'n_jobs': -1
            }
            
            print("Training HIGH-PERFORMANCE model...")
            training_results = api.quick_train_model(session_id, model_config)
            
            # Extract results
            if training_results.get('status') == 'success':
                test_accuracy = training_results.get('evaluation_metrics', {}).get('accuracy')
                validation_accuracy = training_results.get('validation_accuracy')
                
                print(f"Training completed.")
                print(f"  Test Accuracy: {test_accuracy:.4f}")
                if validation_accuracy:
                    print(f"  Validation Accuracy: {validation_accuracy:.4f}")
                
                # Get expected benchmarks from train_rf_clean.py runs
                benchmarks = {
                    'kepler': 0.9460,  # From previous run
                    'k2': 0.9660,      # From current run 
                    'tess': 0.7670     # From current run
                }
                
                expected = benchmarks.get(dataset_name, 0.0)
                print(f"  Expected (train_rf_clean.py): {expected:.4f}")
                print(f"  Difference: {test_accuracy - expected:+.4f}")
                
                if abs(test_accuracy - expected) < 0.02:  # Within 2%
                    print("  ✅ SUCCESS: API matches train_rf_clean.py!")
                else:
                    print("  ❌ GAP: API performance differs from benchmark")
                
                results[dataset_name] = {
                    'success': True,
                    'api_accuracy': test_accuracy,
                    'benchmark_accuracy': expected,
                    'difference': test_accuracy - expected,
                    'features_used': len(all_features),
                    'validation_accuracy': validation_accuracy
                }
                
            else:
                print(f"❌ Training failed: {training_results.get('error', 'Unknown error')}")
                results[dataset_name] = {'success': False, 'error': training_results.get('error')}
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results[dataset_name] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    for dataset, result in results.items():
        print(f"\n{dataset.upper()} Dataset:")
        if result.get('success'):
            print(f"  API Accuracy:       {result['api_accuracy']:.4f}")
            print(f"  Benchmark:          {result['benchmark_accuracy']:.4f}")
            print(f"  Difference:         {result['difference']:+.4f}")
            print(f"  Features Used:      {result['features_used']}")
            if abs(result['difference']) < 0.02:
                print("  Status: ✅ EXCELLENT - Matches benchmark")
            elif abs(result['difference']) < 0.05:
                print("  Status: ✅ GOOD - Close to benchmark")  
            else:
                print("  Status: ❌ NEEDS IMPROVEMENT")
        else:
            print(f"  Status: ❌ FAILED - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_raw_data_performance()