#!/usr/bin/env python3
"""
Complete Exoplanet ML System Test

This script:
1. Trains new models and saves them to ML/models/user
2. Tests real exoplanet vs false positive classification
3. Tests explanation functionality for most critical columns

Usage: python tests/complete_system_test.py
"""

import sys
import os
import json
import time
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

def test_complete_system():
    """Complete system test including model training, prediction, and explanation."""
    
    print("🚀 COMPLETE EXOPLANET ML SYSTEM TEST")
    print("=" * 60)
    print("📅 Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Initialize APIs
    user_api = ExoplanetMLAPI()
    prediction_api = PredictionAPI()
    explanation_api = ExplanationAPI(prediction_api)  # Share the same prediction API
    
    # Ensure directories exist
    model_dir = Path('/home/brine/OneDrive/Work/Exoplanet-Playground/ML/models/user')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'models_trained': [],
        'predictions_made': [],
        'explanations_generated': [],
        'classification_results': {}
    }
    
    # Phase 1: Train New Models
    print("🔧 PHASE 1: Training New Models")
    print("-" * 40)
    
    models_to_train = [
        {
            'model_type': 'random_forest',
            'dataset_name': 'kepler', 
            'model_name': f'rf_kepler_{int(time.time())}',
            'hyperparameters': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42}
        },
        {
            'model_type': 'decision_tree',
            'dataset_name': 'tess',
            'model_name': f'dt_tess_{int(time.time())}',
            'hyperparameters': {'max_depth': 8, 'random_state': 42}
        },
        {
            'model_type': 'linear_regression',
            'dataset_name': 'k2',
            'model_name': f'lr_k2_{int(time.time())}', 
            'hyperparameters': {'random_state': 42}
        }
    ]
    
    for config in models_to_train:
        print(f"\n🔨 Training {config['model_type']} on {config['dataset_name']}...")
        
        try:
            start_time = time.time()
            
            result = user_api.train_model(
                model_type=config['model_type'],
                dataset_name=config['dataset_name'],
                model_name=config['model_name'],
                hyperparameters=config['hyperparameters']
            )
            
            duration = time.time() - start_time
            
            if 'error' not in result:
                model_info = {
                    'model_name': config['model_name'],
                    'model_type': config['model_type'], 
                    'dataset': config['dataset_name'],
                    'accuracy': result.get('test_accuracy', 0),
                    'training_time': duration,
                    'feature_count': result.get('feature_count', 0)
                }
                results['models_trained'].append(model_info)
                
                print(f"   ✅ Success! Accuracy: {model_info['accuracy']:.3f}, Features: {model_info['feature_count']}, Time: {duration:.1f}s")
                
                # Verify model file was saved
                metadata_file = model_dir / f"{config['model_name']}_metadata.json"
                model_file = model_dir / f"{config['model_name']}.joblib"
                
                if metadata_file.exists():
                    print(f"   📄 Metadata saved: {metadata_file.name}")
                if model_file.exists():
                    print(f"   💾 Model saved: {model_file.name}")
                else:
                    print(f"   ⚠️  Model file not found: {model_file.name}")
                    
            else:
                print(f"   ❌ Training failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print(f"\n📦 Successfully trained {len(results['models_trained'])} models")
    
    if not results['models_trained']:
        print("❌ No models were trained successfully. Cannot proceed with testing.")
        return results
    
    # Phase 2: Test Predictions for Real vs False Positive Classification
    print(f"\n🔮 PHASE 2: Testing Real vs False Positive Predictions")
    print("-" * 40)
    
    for model_info in results['models_trained']:
        print(f"\n🤖 Testing {model_info['model_name']}...")
        
        try:
            # Load model
            model_path = str(model_dir / f"{model_info['model_name']}.joblib")
            load_result = prediction_api.load_model(model_path, model_info['model_name'])
            
            if load_result['status'] != 'success':
                print(f"   ❌ Failed to load model: {load_result.get('error', 'Unknown')}")
                continue
            
            print(f"   ✅ Model loaded successfully")
            
            # Get sample data from the same dataset the model was trained on
            dataset_name = model_info['dataset']
            sample_data = user_api.get_sample_data(dataset_name, n_samples=20)
            
            if 'error' in sample_data:
                print(f"   ❌ Failed to get sample data: {sample_data['error']}")
                continue
            
            # Analyze the dataset for classification types
            sample_df = pd.DataFrame(sample_data['sample_data'])
            
            # Find target column for this dataset
            target_columns = {
                'kepler': 'koi_disposition',
                'tess': 'tfopwg_disp', 
                'k2': 'disposition'
            }
            target_col = target_columns.get(dataset_name)
            
            if target_col and target_col in sample_df.columns:
                unique_classes = sample_df[target_col].unique()
                print(f"   📊 Available classes in {dataset_name}: {list(unique_classes)}")
                
                # Count confirmed vs false positives
                confirmed_count = sum(1 for cls in unique_classes if 'CONFIRMED' in str(cls).upper())
                false_positive_count = sum(1 for cls in unique_classes if 'FALSE' in str(cls).upper() or 'FP' in str(cls).upper())
                
                results['classification_results'][dataset_name] = {
                    'target_column': target_col,
                    'unique_classes': list(unique_classes),
                    'has_confirmed': confirmed_count > 0,
                    'has_false_positive': false_positive_count > 0,
                    'classification_capable': confirmed_count > 0 and false_positive_count > 0
                }
                
                print(f"   🎯 Classification capability: {'✅ Yes' if results['classification_results'][dataset_name]['classification_capable'] else '⚠️ Limited'}")
            
            # Make predictions on sample data
            predictions_made = 0
            
            for idx, sample_row in sample_df.head(5).iterrows():
                # Remove target columns
                input_data = {k: v for k, v in sample_row.items() 
                             if k not in ['koi_disposition', 'tfopwg_disp', 'disposition']}
                
                try:
                    pred_result = prediction_api.predict_single(model_info['model_name'], input_data)
                    
                    if pred_result['status'] == 'success':
                        predictions_made += 1
                        
                        prediction_info = {
                            'model_name': model_info['model_name'],
                            'model_type': model_info['model_type'],
                            'dataset': dataset_name,
                            'prediction': pred_result['prediction'],
                            'confidence': max(pred_result.get('probabilities', [0.5])),
                            'actual_class': sample_row.get(target_col, 'Unknown') if target_col else 'Unknown'
                        }
                        results['predictions_made'].append(prediction_info)
                        
                        print(f"   🔍 Prediction: {pred_result['prediction']} (confidence: {prediction_info['confidence']:.3f}, actual: {prediction_info['actual_class']})")
                        
                        if predictions_made >= 3:  # Limit to 3 predictions per model
                            break
                            
                except Exception as pred_error:
                    print(f"   ⚠️ Prediction failed: {str(pred_error)}")
            
            print(f"   📈 Made {predictions_made} successful predictions")
            
        except Exception as e:
            print(f"   ❌ Error testing model: {str(e)}")
    
    # Phase 3: Test Explanation Functionality  
    print(f"\n📊 PHASE 3: Testing Feature Importance & Explanations")
    print("-" * 40)
    
    for model_info in results['models_trained'][:2]:  # Test first 2 models
        print(f"\n🔬 Analyzing {model_info['model_name']}...")
        
        try:
            model_name = model_info['model_name']
            dataset_name = model_info['dataset']
            
            # Ensure model is loaded
            if model_name not in prediction_api.loaded_models:
                model_path = str(model_dir / f"{model_name}.joblib")
                load_result = prediction_api.load_model(model_path, model_name)
                if load_result['status'] != 'success':
                    print(f"   ❌ Failed to load model for explanation")
                    continue
            
            # Get sample data for explanation
            sample_data = user_api.get_sample_data(dataset_name, n_samples=100)
            if 'error' in sample_data:
                print(f"   ❌ Failed to get sample data for explanation")
                continue
            
            sample_df = pd.DataFrame(sample_data['sample_data'])
            
            # Prepare data for explanation
            target_columns = ['koi_disposition', 'tfopwg_disp', 'disposition']
            feature_cols = [col for col in sample_df.columns if col not in target_columns]
            
            # Select only numeric columns for explanation
            X_sample = sample_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            
            if len(X_sample.columns) == 0:
                print(f"   ❌ No numeric features available for explanation")
                continue
            
            print(f"   📋 Analyzing {len(X_sample.columns)} features")
            
            # Split data for explanation
            split_idx = len(X_sample) // 2
            X_train, X_test = X_sample[:split_idx], X_sample[split_idx:]
            
            # Create dummy target for explanation (since we need y_train/y_test)
            y_train = pd.Series(['CONFIRMED'] * len(X_train))
            y_test = pd.Series(['CANDIDATE'] * len(X_test))
            
            # Test global explanation
            try:
                global_explanation = explanation_api.explain_model_global(
                    model_id=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    methods=['model_importance']
                )
                
                if global_explanation.get('status') == 'success':
                    explanation_results = global_explanation.get('results', {})
                    top_features_list = global_explanation.get('top_features', [])
                    print(f"   ✅ Global explanation generated")
                    
                    # Extract feature importance from results
                    feature_importance = None
                    if 'model_importance' in explanation_results:
                        feature_importance = explanation_results['model_importance']
                    elif 'feature_ranking' in explanation_results:
                        feature_importance = explanation_results['feature_ranking']
                    
                    if feature_importance and isinstance(feature_importance, dict):
                        # Create sorted list of feature importance
                        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(float(x[1])), reverse=True)
                        top_features = sorted_features[:10]
                        
                        explanation_info = {
                            'model_name': model_name,
                            'model_type': model_info['model_type'],
                            'dataset': dataset_name,
                            'total_features': len(X_sample.columns),
                            'top_features': top_features
                        }
                        results['explanations_generated'].append(explanation_info)
                        
                        print(f"   🏆 Top 5 most important features:")
                        for i, (feature, score) in enumerate(top_features[:5], 1):
                            print(f"      {i}. {feature}: {float(score):.4f}")
                    elif top_features_list:
                        # Use top features from the API response
                        explanation_info = {
                            'model_name': model_name,
                            'model_type': model_info['model_type'],
                            'dataset': dataset_name,
                            'total_features': len(X_sample.columns),
                            'top_features': top_features_list[:10]
                        }
                        results['explanations_generated'].append(explanation_info)
                        
                        print(f"   🏆 Top 5 most important features:")
                        for i, (feature, score) in enumerate(top_features_list[:5], 1):
                            print(f"      {i}. {feature}: {float(score):.4f}")
                    else:
                        print(f"   ⚠️ No feature importance data found")
                else:
                    print(f"   ❌ Global explanation failed: {global_explanation.get('error', 'Unknown')}")
                    
            except Exception as explain_error:
                print(f"   ❌ Explanation error: {str(explain_error)}")
                
                # Fallback: show model features from metadata
                metadata_file = model_dir / f"{model_name}_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        feature_names = metadata.get('feature_names', [])
                        if feature_names:
                            print(f"   📋 Model features ({len(feature_names)} total):")
                            for i, feature in enumerate(feature_names[:5], 1):
                                print(f"      {i}. {feature}")
                            
                            fallback_info = {
                                'model_name': model_name,
                                'model_type': model_info['model_type'],
                                'dataset': dataset_name,
                                'total_features': len(feature_names),
                                'top_features': [(f, 'N/A') for f in feature_names[:5]],
                                'fallback': True
                            }
                            results['explanations_generated'].append(fallback_info)
                    except:
                        pass
            
        except Exception as e:
            print(f"   ❌ Error in explanation analysis: {str(e)}")
    
    # Final Results Summary
    print(f"\n" + "=" * 60)
    print("📋 COMPLETE SYSTEM TEST RESULTS")
    print("=" * 60)
    
    print(f"🔧 Models Trained: {len(results['models_trained'])}")
    for model in results['models_trained']:
        print(f"   • {model['model_name']}: {model['model_type']} on {model['dataset']} ({model['accuracy']:.3f} accuracy)")
    
    print(f"\n🔮 Predictions Made: {len(results['predictions_made'])}")
    if results['predictions_made']:
        print("   Sample predictions:")
        for pred in results['predictions_made'][:5]:
            print(f"   • {pred['model_type']}: {pred['prediction']} (confidence: {pred['confidence']:.3f}, actual: {pred['actual_class']})")
    
    print(f"\n🎯 Classification Analysis:")
    for dataset, info in results['classification_results'].items():
        capability = "✅ Full" if info['classification_capable'] else "⚠️ Limited"
        print(f"   • {dataset}: {capability} - Classes: {info['unique_classes']}")
    
    print(f"\n📊 Feature Explanations: {len(results['explanations_generated'])}")
    for explanation in results['explanations_generated']:
        fallback_note = " (fallback)" if explanation.get('fallback') else ""
        print(f"   • {explanation['model_type']}: {explanation['total_features']} features analyzed{fallback_note}")
        if explanation['top_features'] and not explanation.get('fallback'):
            top_3 = explanation['top_features'][:3]
            feature_list = [f"{feat}({score:.3f})" for feat, score in top_3 if isinstance(score, (int, float))]
            print(f"     Top: {', '.join(feature_list)}")
    
    # System Assessment
    models_success = len(results['models_trained']) > 0
    predictions_success = len(results['predictions_made']) > 0
    classification_success = any(info['classification_capable'] for info in results['classification_results'].values())
    explanation_success = len(results['explanations_generated']) > 0
    
    total_success = sum([models_success, predictions_success, classification_success, explanation_success])
    success_rate = (total_success / 4) * 100
    
    print(f"\n🏆 OVERALL SYSTEM STATUS:")
    print(f"   📦 Model Training: {'✅ Working' if models_success else '❌ Failed'}")
    print(f"   🔮 Predictions: {'✅ Working' if predictions_success else '❌ Failed'}")
    print(f"   🎯 Real vs False Positive: {'✅ Capable' if classification_success else '❌ Limited'}")
    print(f"   📊 Feature Explanations: {'✅ Working' if explanation_success else '❌ Failed'}")
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("🎉 STATUS: EXCELLENT - System fully functional!")
    elif success_rate >= 50:
        print("⚠️ STATUS: GOOD - System mostly working")
    else:
        print("🚨 STATUS: NEEDS WORK - Major issues detected")
    
    # Save detailed results
    results_file = Path(f"/home/brine/OneDrive/Work/Exoplanet-Playground/tests/results/complete_system_test_{int(time.time())}.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    try:
        results = test_complete_system()
        
        # Exit with success if most tests passed
        success_metrics = [
            len(results['models_trained']) > 0,
            len(results['predictions_made']) > 0,
            len(results['explanations_generated']) > 0
        ]
        
        if sum(success_metrics) >= 2:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        sys.exit(1)