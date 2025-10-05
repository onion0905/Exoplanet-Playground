#!/usr/bin/env python3
"""
Simple test to verify 90%+ accuracy and confusion matrix using direct pandas loading
"""

import pandas as pd
import numpy as np
import sys
import os
import time

# Add ML directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ML', 'src'))

from ML.src.api.training_api import TrainingAPI

def load_csv_with_comments(data_path):
    """Load CSV file, handling comment lines like train_rf_clean.py does."""
    print(f"Loading data from: {data_path}")
    
    # Skip comment lines that start with #
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    # Find first non-comment line for header
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith('#') and line.strip():
            header_idx = i
            break
    
    # Load the CSV
    df = pd.read_csv(data_path, skiprows=header_idx)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def test_dataset_accuracy(dataset_name, data_path, target_col):
    """Test a dataset for 90%+ accuracy and show confusion matrix."""
    
    print(f"\n{'='*60}")
    print(f"TESTING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Load data using the same method as train_rf_clean.py
    data = load_csv_with_comments(data_path)
    
    # Initialize API
    api = TrainingAPI()
    
    # Quick configure training with optimized settings
    config = {
        'target_column': target_col,
        'preprocessing_config': {
            'scale_features': False,
            'max_missing_ratio': 0.999,
            'missing_strategy': 'median'
        }
    }
    
    session_id = api.quick_configure_training(data, config)
    print(f"Training session configured: {session_id}")
    
    # Train with high-performance Random Forest settings
    model_config = {
        'model_type': 'random_forest',
        'n_estimators': 300,
        'max_depth': 20,
        'random_state': 42
    }
    
    start_time = time.time()
    result = api.quick_train_model(session_id, model_config)
    training_time = time.time() - start_time
    
    # Extract metrics
    metrics = result.get('metrics', {})
    accuracy = metrics.get('accuracy', 0.0)
    val_accuracy = metrics.get('val_accuracy', 0.0)
    feature_count = result.get('feature_count', 0)
    
    print(f"\n🎯 RESULTS:")
    print(f"  Training Time: {training_time:.2f} seconds")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Features Used: {feature_count}")
    
    # Check 90%+ achievement
    achieved_90_plus = accuracy >= 0.90
    status_emoji = "✅" if achieved_90_plus else "❌"
    print(f"  90%+ Target: {status_emoji} {'ACHIEVED' if achieved_90_plus else 'NOT ACHIEVED'}")
    
    # Show confusion matrix if available
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        class_names = metrics.get('class_names', ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
        
        print(f"\n📊 CONFUSION MATRIX:")
        print(f"                    Predicted")
        header = "Actual       " + " ".join(f"{name[:4]:>6s}" for name in class_names)
        print(header)
        print("-" * len(header))
        
        total_correct = 0
        total_samples = 0
        
        for i, actual_class in enumerate(class_names):
            row = f"{actual_class[:10]:12s} "
            row += " ".join(f"{cm[i][j]:6d}" for j in range(len(cm[i])))
            print(row)
            
            total_correct += cm[i][i]
            total_samples += sum(cm[i])
        
        manual_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"\nManual accuracy check: {manual_accuracy:.4f} ({manual_accuracy*100:.2f}%)")
    else:
        print(f"\n⚠️ No confusion matrix in results")
    
    return {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'feature_count': feature_count,
        'achieved_90_plus': achieved_90_plus,
        'training_time': training_time
    }

def main():
    """Test all datasets for 90%+ accuracy."""
    
    print("🚀 ACCURACY & CONFUSION MATRIX VERIFICATION")
    print("=" * 60)
    
    datasets = [
        ('kepler', 'data/kepler_raw.csv', 'koi_disposition'),
        ('k2', 'data/k2_raw.csv', 'disposition'),
        ('tess', 'data/tess_raw.csv', 'tfopwg_disp')
    ]
    
    results = []
    
    for dataset_name, data_path, target_col in datasets:
        try:
            result = test_dataset_accuracy(dataset_name, data_path, target_col)
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'dataset': dataset_name,
                'accuracy': 0.0,
                'achieved_90_plus': False,
                'error': str(e)
            })
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏆 FINAL SUMMARY")
    print(f"{'='*60}")
    
    datasets_90_plus = 0
    benchmarks = {'kepler': 0.946, 'k2': 0.966, 'tess': 0.767}
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['dataset'].upper():8s}: ERROR - {result['error']}")
            continue
            
        accuracy = result['accuracy']
        feature_count = result['feature_count']
        achieved = result['achieved_90_plus']
        
        status_emoji = "✅" if achieved else ("⚠️" if accuracy >= 0.75 else "❌")
        benchmark = benchmarks.get(result['dataset'], 0)
        diff = accuracy - benchmark
        
        print(f"{status_emoji} {result['dataset'].upper():8s}: {accuracy:.3f} ({accuracy*100:.1f}%) - {feature_count} features")
        print(f"    vs benchmark {benchmark:.3f} (diff: {diff:+.3f})")
        
        if achieved:
            datasets_90_plus += 1
    
    print(f"\n🎯 DATASETS ACHIEVING 90%+: {datasets_90_plus}/3")
    
    if datasets_90_plus >= 2:
        print("🎉 SUCCESS: Multiple datasets achieved 90%+ accuracy!")
    elif datasets_90_plus == 1:
        print("⚠️  PARTIAL SUCCESS: One dataset achieved 90%+ accuracy")
    else:
        print("❌ FAILURE: No datasets achieved 90%+ accuracy")

if __name__ == "__main__":
    main()