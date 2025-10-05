#!/usr/bin/env python3
"""
Test script for NASA API column filtering integration.
Tests the ML API with the new column filteri    for dataset_type, result in results.items():
        print(f"\n{dataset_type.upper()} Dataset:")
        if result['success']:
            print(f"  ✓ Success")
            print(f"  Original columns: {result['original_columns']}")
            print(f"  Filtered columns: {result['filtered_columns']}")
            print(f"  Excluded columns: {result['excluded_count']}")
            print(f"  Feature columns: {result['feature_count']}")
            if result.get('test_accuracy') is not None:
                print(f"  Test accuracy: {result['test_accuracy']:.4f}")
            if result.get('validation_accuracy') is not None:
                print(f"  Validation accuracy: {result['validation_accuracy']:.4f}")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")"""

import sys
import os
sys.path.append('/home/brine/OneDrive/Work/Exoplanet-Playground')

from ML.src.api.training_api import TrainingAPI
from ML.src.data.data_loader import DataLoader
import pandas as pd
import json


def test_nasa_api_filtering():
    """Test NASA API column filtering across all exoplanet datasets using RAW data for best performance."""
    
    # Datasets to test - use RAW data like train_rf_clean.py for maximum performance
    datasets = {
        'kepler': 'data/kepler_raw.csv',
        'k2': 'data/k2_raw.csv', 
        'tess': 'data/tess_raw.csv'
    }
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Testing {dataset_name.upper()} Dataset")
        print(f"{'='*60}")
        
        try:
            # Initialize API
            api = TrainingAPI()
            
            # Load RAW data directly (same as train_rf_clean.py for maximum performance)
            df_path = datasets[dataset_name]
            print(f"Loading raw data from: {df_path}")
            df = pd.read_csv(df_path, comment='#')
            
            print(f"Original data shape: {df.shape}")
            print(f"Original columns: {len(df.columns)}")
            
            # Apply NASA API filtering
            df_filtered, excluded_cols = api.data_processor.apply_nasa_api_filtering(df)
            
            print(f"Filtered data shape: {df_filtered.shape}")
            print(f"Excluded columns ({len(excluded_cols)}): {excluded_cols}")
            
            # Detect target column
            detected_type, target_col = api.data_processor.column_filter.detect_dataset_type(df_filtered)
            print(f"Detected dataset type: {detected_type}")
            print(f"Target column: {target_col}")
            
            # Get feature columns
            feature_cols = api.data_processor.column_filter.get_feature_columns(df_filtered)
            print(f"Feature columns: {len(feature_cols)}")
            
            # Check target distribution
            if target_col in df_filtered.columns:
                target_series = df_filtered[target_col]
                # Apply label mapping
                target_mapped = api.data_processor.column_filter.map_target_labels(target_series, detected_type)
                print(f"\nTarget distribution:")
                print(target_mapped.value_counts())
            
            # Try a small training run to test the full pipeline
            print(f"\nTesting training pipeline...")
            
            # Configure training with NASA API filtering
            config = {
                'target_column': target_col,
                'feature_columns': feature_cols,  # Use all available features for best performance
                'model_type': 'random_forest',
                'use_nasa_filtering': False,  # Already filtered above
                'preprocessing_config': {
                    'max_missing_ratio': 0.999,  # Only exclude columns with >99.9% missing
                    'handle_missing': True,
                    'missing_strategy': 'median',  # Same as train_rf_clean.py
                    'scale_features': False,       # Don't scale (RF works better without)
                    'encode_categorical': True,
                    'remove_outliers': False
                }
            }
            
            session_id = api.quick_configure_training(df_filtered, config)
            print(f"Training configured. Session ID: {session_id}")
            
            # Train model with high-performance settings
            model_config = {
                'model_type': 'random_forest',
                'n_estimators': 300,  # High-performance settings to match train_rf_clean.py
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }
            
            training_results = api.quick_train_model(session_id, model_config)
            
            # Extract accuracy metrics from results
            test_accuracy = None
            validation_accuracy = training_results.get('validation_accuracy')
            if training_results.get('status') == 'success' and 'evaluation_metrics' in training_results:
                test_accuracy = training_results['evaluation_metrics'].get('accuracy')
            
            print(f"Training completed.")
            if test_accuracy is not None:
                print(f"  Test Accuracy: {test_accuracy:.4f}")
            if validation_accuracy is not None:
                print(f"  Validation Accuracy: {validation_accuracy:.4f}")
            
            results[dataset_name] = {
                'success': True,
                'original_columns': len(df.columns),
                'filtered_columns': len(df_filtered.columns),
                'excluded_count': len(excluded_cols),
                'target_column': target_col,
                'feature_count': len(feature_cols),
                'test_accuracy': test_accuracy,
                'validation_accuracy': validation_accuracy
            }
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results[dataset_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF NASA API FILTERING TESTS")
    print(f"{'='*60}")
    
    for dataset_type, result in results.items():
        print(f"\n{dataset_type.upper()} Dataset:")
        if result['success']:
            print(f"  ✓ Success")
            print(f"  Original columns: {result['original_columns']}")
            print(f"  Filtered columns: {result['filtered_columns']}")
            print(f"  Excluded columns: {result['excluded_count']}")
            print(f"  Feature columns: {result['feature_count']}")
            if result.get('test_accuracy') is not None:
                print(f"  Test accuracy: {result['test_accuracy']:.4f}")
            if result.get('validation_accuracy') is not None:
                print(f"  Validation accuracy: {result['validation_accuracy']:.4f}")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    return results


if __name__ == "__main__":
    test_nasa_api_filtering()