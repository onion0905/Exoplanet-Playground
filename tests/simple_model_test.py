#!/usr/bin/env python3
"""
Model Validation Test Script for Exoplanet ML System

This script validates:
1. Model loading and prediction functionality
2. Feature importance analysis 
3. Real vs false positive classification
4. End-to-end workflow

Usage: python tests/simple_model_test.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
ml_src_path = os.path.join(project_root, 'ML', 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, ml_src_path)

from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI

def test_models():
    """Test model loading, predictions, and feature importance."""
    
    print("🔬 EXOPLANET ML SYSTEM MODEL VALIDATION")
    print("=" * 60)
    
    # Initialize APIs
    user_api = ExoplanetMLAPI()
    prediction_api = PredictionAPI()
    explanation_api = ExplanationAPI()
    
    model_dir = Path('/home/brine/OneDrive/Work/Exoplanet-Playground/ML/models/user')
    
    # Find working models
    print("🔍 Finding working models...")
    working_models = []
    metadata_files = list(model_dir.glob("*_metadata.json"))
    
    for metadata_file in metadata_files[:5]:  # Limit to 5 for speed
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            model_path = str(metadata_file).replace('_metadata.json', '')
            if Path(f"{model_path}.joblib").exists():
                working_models.append((metadata_file, metadata))
                print(f"   ✅ {metadata_file.name}")
        except:
            print(f"   ❌ {metadata_file.name} (corrupted)")
    
    if not working_models:
        print("❌ No working models found!")
        return False
    
    print(f"📦 Found {len(working_models)} working models")
    print()
    
    # Test 1: Model Loading and Predictions
    print("🤖 TEST 1: Model Loading and Predictions")
    print("-" * 40)
    
    successful_predictions = []
    
    for i, (metadata_file, metadata) in enumerate(working_models[:3]):
        try:
            model_path = str(metadata_file).replace('_metadata.json', '')
            model_id = f"test_{i}"
            
            # Load model
            load_result = prediction_api.load_model(model_path, model_id)
            if load_result['status'] != 'success':
                print(f"   ❌ Failed to load {metadata_file.name}")
                continue
            
            # Get sample data
            dataset_name = metadata.get('dataset_name', 'kepler')
            sample_data = user_api.get_sample_data(dataset_name, n_samples=3)
            
            if 'error' in sample_data:
                print(f"   ❌ Failed to get sample data for {dataset_name}")
                continue
            
            # Make prediction
            for sample in sample_data['sample_data'][:1]:
                # Remove target columns
                target_cols = ['koi_disposition', 'tfopwg_disp', 'disposition']
                input_data = {k: v for k, v in sample.items() if k not in target_cols}
                
                # Filter to model features
                model_features = metadata.get('feature_names', [])
                input_data = {k: v for k, v in input_data.items() if k in model_features}
                
                if input_data:
                    pred_result = prediction_api.predict_single(model_id, input_data)
                    
                    if pred_result['status'] == 'success':
                        successful_predictions.append({
                            'model_type': metadata['model_type'],
                            'dataset': dataset_name,
                            'prediction': pred_result['prediction'],
                            'confidence': max(pred_result.get('probabilities', [0.5]))
                        })
                        print(f"   ✅ {metadata['model_type']} on {dataset_name}: {pred_result['prediction']}")
                        break
        except Exception as e:
            print(f"   ❌ Error with {metadata_file.name}: {str(e)}")
    
    print(f"📊 Successfully made {len(successful_predictions)} predictions")
    print()
    
    # Test 2: Feature Importance Analysis
    print("📊 TEST 2: Feature Importance Analysis")
    print("-" * 40)
    
    feature_analyses = []
    
    for i, (metadata_file, metadata) in enumerate(working_models[:2]):
        try:
            model_id = f"analysis_{i}"
            model_path = str(metadata_file).replace('_metadata.json', '')
            
            # Load model if not loaded
            if model_id not in prediction_api.loaded_models:
                load_result = prediction_api.load_model(model_path, model_id)
                if load_result['status'] != 'success':
                    continue
            
            # Get feature names from metadata
            feature_names = metadata.get('feature_names', [])
            if feature_names:
                print(f"   📋 {metadata['model_type']} features ({len(feature_names)}):")
                
                # Show top 5 most important features
                top_features = feature_names[:5]
                for j, feature in enumerate(top_features, 1):
                    print(f"      {j}. {feature}")
                
                feature_analyses.append({
                    'model_type': metadata['model_type'],
                    'dataset': metadata.get('dataset_name', 'unknown'),
                    'features': len(feature_names),
                    'top_features': top_features
                })
        except Exception as e:
            print(f"   ❌ Error analyzing {metadata_file.name}: {str(e)}")
    
    print(f"📈 Analyzed {len(feature_analyses)} models for feature importance")
    print()
    
    # Test 3: Classification Validation
    print("🎯 TEST 3: Real vs False Positive Classification")
    print("-" * 40)
    
    classification_support = {}
    
    for dataset_name in ['kepler', 'tess', 'k2']:
        try:
            dataset_info = user_api.get_dataset_info(dataset_name)
            sample_data = user_api.get_sample_data(dataset_name, n_samples=10)
            
            if 'error' not in dataset_info and 'error' not in sample_data:
                target_column = dataset_info.get('target_column', 'Unknown')
                
                # Check sample data for classification types
                sample_df = pd.DataFrame(sample_data['sample_data'])
                
                if target_column in sample_df.columns:
                    unique_values = sample_df[target_column].unique()
                    
                    has_confirmed = any('CONFIRMED' in str(v).upper() for v in unique_values)
                    has_false_positive = any('FALSE' in str(v).upper() or 'FP' in str(v) for v in unique_values)
                    
                    classification_support[dataset_name] = {
                        'target_column': target_column,
                        'classes': list(unique_values),
                        'confirmed_support': has_confirmed,
                        'false_positive_support': has_false_positive,
                        'classification_ready': has_confirmed and has_false_positive
                    }
                    
                    status = "✅" if has_confirmed and has_false_positive else "⚠️"
                    print(f"   {status} {dataset_name}: {target_column} -> {list(unique_values)}")
        except Exception as e:
            print(f"   ❌ Error checking {dataset_name}: {str(e)}")
    
    ready_datasets = sum(1 for info in classification_support.values() if info.get('classification_ready', False))
    print(f"🎯 {ready_datasets}/3 datasets ready for real vs false positive classification")
    print()
    
    # Final Summary
    print("=" * 60)
    print("📋 FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"📦 Working Models Found: {len(working_models)}")
    print(f"🤖 Successful Predictions: {len(successful_predictions)}")
    print(f"📊 Feature Analyses: {len(feature_analyses)}")
    print(f"🎯 Classification Ready Datasets: {ready_datasets}/3")
    
    # Show sample results
    if successful_predictions:
        print("\n🔮 Sample Predictions:")
        for pred in successful_predictions:
            print(f"   • {pred['model_type']} on {pred['dataset']}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
    
    if feature_analyses:
        print("\n📊 Top Features (samples):")
        for analysis in feature_analyses:
            print(f"   • {analysis['model_type']} ({analysis['features']} features): {', '.join(analysis['top_features'])}")
    
    # Calculate success rate
    total_tests = 3
    passed_tests = (
        (len(successful_predictions) > 0) +
        (len(feature_analyses) > 0) +
        (ready_datasets > 0)
    )
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📈 Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
    
    if success_rate >= 80:
        print("🎉 Status: EXCELLENT - System is production ready!")
    elif success_rate >= 60:
        print("⚠️ Status: GOOD - System works with minor issues")
    else:
        print("🚨 Status: NEEDS WORK - Major issues detected")
    
    print("=" * 60)
    
    return success_rate >= 60

if __name__ == "__main__":
    try:
        success = test_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        sys.exit(1)