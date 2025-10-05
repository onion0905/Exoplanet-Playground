#!/usr/bin/env python3
"""
Test accuracy verification and confusion matrix display using the Training API
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Add ML directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ML', 'src'))

from ML.src.api.training_api import TrainingAPI
from ML.src.data.data_loader import DataLoader
from sklearn.metrics import confusion_matrix

def test_dataset_with_confusion_matrix(dataset_name, data_path, target_col):
    """Test a dataset and display detailed results including confusion matrix."""
    
    print(f"\n{'='*60}")
    print(f"TESTING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Load data
    loader = DataLoader()
    data = loader.load_user_dataset(data_path)
    print(f"Loaded {len(data)} rows and {len(data.columns)} columns")
    
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
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Model ID: {result.get('model_id', 'N/A')}")
    
    # Extract metrics
    metrics = result.get('metrics', {})
    accuracy = metrics.get('accuracy', 0.0)
    val_accuracy = metrics.get('val_accuracy', 0.0)
    feature_count = result.get('feature_count', 0)
    
    print(f"\nPerformance Metrics:")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Features Used: {feature_count}")
    
    # Check if we achieved 90%+ accuracy
    achieved_90_plus = accuracy >= 0.90
    print(f"  90%+ Accuracy Achieved: {'✓ YES' if achieved_90_plus else '✗ NO'}")
    
    # Display confusion matrix if available
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        class_names = metrics.get('class_names', ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
        
        print(f"\nConfusion Matrix:")
        print(f"                    Predicted")
        header = "Actual       " + " ".join(f"{name[:4]:>6s}" for name in class_names)
        print(header)
        
        for i, actual_class in enumerate(class_names):
            row = f"{actual_class[:8]:12s} "
            row += " ".join(f"{cm[i][j]:6d}" for j in range(len(cm[i])))
            print(row)
            
        # Calculate per-class metrics
        print(f"\nPer-Class Performance:")
        for i, class_name in enumerate(class_names):
            tp = cm[i][i]  # True positives
            fp = sum(cm[j][i] for j in range(len(cm)) if j != i)  # False positives
            fn = sum(cm[i][j] for j in range(len(cm[i])) if j != i)  # False negatives
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {class_name:15s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'feature_count': feature_count,
        'achieved_90_plus': achieved_90_plus,
        'training_time': training_time
    }

def main():
    """Test all three datasets."""
    
    print("🚀 TESTING 90%+ ACCURACY WITH CONFUSION MATRIX")
    print("=" * 60)
    
    datasets = [
        ('kepler', 'data/kepler_raw.csv', 'koi_disposition'),
        ('k2', 'data/k2_raw.csv', 'disposition'), 
        ('tess', 'data/tess_raw.csv', 'tfopwg_disp')
    ]
    
    results = []
    
    for dataset_name, data_path, target_col in datasets:
        try:
            result = test_dataset_with_confusion_matrix(dataset_name, data_path, target_col)
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {dataset_name}: {e}")
            results.append({
                'dataset': dataset_name,
                'accuracy': 0.0,
                'achieved_90_plus': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    datasets_90_plus = 0
    for result in results:
        status = "✓ PASS" if result.get('achieved_90_plus', False) else "✗ FAIL"
        accuracy = result.get('accuracy', 0.0)
        feature_count = result.get('feature_count', 0)
        
        print(f"{result['dataset'].upper():8s}: {accuracy:.3f} ({accuracy*100:.1f}%) - {feature_count} features - {status}")
        
        if result.get('achieved_90_plus', False):
            datasets_90_plus += 1
    
    print(f"\nDatasets achieving 90%+ accuracy: {datasets_90_plus}/3")
    
    # Verification against benchmarks
    benchmarks = {
        'kepler': 0.946,  # 94.6%
        'k2': 0.966,      # 96.6% 
        'tess': 0.767     # 76.7%
    }
    
    print(f"\nBenchmark Comparison:")
    for result in results:
        dataset = result['dataset']
        if dataset in benchmarks:
            benchmark = benchmarks[dataset]
            actual = result.get('accuracy', 0.0)
            diff = actual - benchmark
            status = "✓" if abs(diff) <= 0.02 else "⚠"
            print(f"  {dataset.upper():8s}: {actual:.3f} vs {benchmark:.3f} benchmark (diff: {diff:+.3f}) {status}")

if __name__ == "__main__":
    main()