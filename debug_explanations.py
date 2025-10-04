#!/usr/bin/env python3
"""
Test Feature Explanations Debug
Debug what structure is returned by the explanation API
"""

import sys
import os
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
ml_src_path = project_root / 'ML' / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(ml_src_path))

from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI
import time
import pandas as pd

def debug_explanations():
    """Debug what explanations look like."""
    print("🔍 DEBUGGING FEATURE EXPLANATIONS")
    print("=" * 50)
    
    # Initialize APIs
    user_api = ExoplanetMLAPI()
    prediction_api = PredictionAPI()
    explanation_api = ExplanationAPI(prediction_api)
    
    # Train a model
    result = user_api.train_model("random_forest", "kepler", f"debug_explain_{int(time.time())}")
    model_name = result['model_name']
    
    print(f"✅ Trained model: {model_name}")
    
    # Load model for prediction
    model_path = user_api.user_dir / f"{model_name}.joblib"
    load_result = prediction_api.load_model(str(model_path), model_name)
    
    if load_result['status'] != 'success':
        print(f"❌ Failed to load model: {load_result}")
        return
        
    print("✅ Model loaded for prediction/explanation")
    
    # Get sample data  
    sample_data = user_api.get_sample_data("kepler", n_samples=100)
    sample_df = pd.DataFrame(sample_data['sample_data'])
    
    # Prepare data for explanation
    X_sample = sample_df.drop(columns=['koi_disposition'], errors='ignore')
    split_idx = len(X_sample) // 2
    X_train, X_test = X_sample[:split_idx], X_sample[split_idx:]
    
    # Create dummy target for explanation
    y_train = pd.Series(['CONFIRMED'] * len(X_train))
    y_test = pd.Series(['CANDIDATE'] * len(X_test))
    
    print(f"📊 Prepared data: {len(X_train)} train, {len(X_test)} test samples")
    
    # Test explanation
    explanation = explanation_api.explain_model_global(
        model_id=model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        methods=['model_importance']
    )
    
    print("\n🔬 EXPLANATION RESULTS:")
    print(f"Status: {explanation.get('status')}")
    if 'error' in explanation:
        print(f"Error: {explanation['error']}")
        return
        
    results = explanation.get('results', {})
    print(f"Results keys: {list(results.keys())}")
    
    for method, method_results in results.items():
        print(f"\n📋 Method: {method}")
        print(f"  Type: {type(method_results)}")
        if isinstance(method_results, dict):
            print(f"  Keys: {list(method_results.keys())}")
            for key, value in method_results.items():
                if isinstance(value, (list, dict)):
                    print(f"    {key}: {type(value)} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
                else:
                    print(f"    {key}: {value}")
        
    # Check top features
    top_features = explanation.get('top_features', {})
    print(f"\n🏆 Top Features: {top_features}")

if __name__ == "__main__":
    debug_explanations()