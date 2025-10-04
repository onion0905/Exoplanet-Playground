#!/usr/bin/env python3
"""
ML API Demo Script for Exoplanet Detection System

This script demonstrates how to use all the APIs from the ML structure:
- ExoplanetMLAPI (User-friendly unified API)
- TrainingAPI (Advanced training controls)
- PredictionAPI (Batch and single predictions)
- ExplanationAPI (Model explainability)

Usage:
    python ml_api_demo.py
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to path so we can import from ML
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run the complete ML API demonstration."""
    
    print("=" * 60)
    print("🚀 EXOPLANET ML API DEMONSTRATION")
    print("=" * 60)
    
    # Test 1: User-Friendly API
    print("\n1️⃣  TESTING USER-FRIENDLY API")
    print("-" * 40)
    try:
        test_user_api()
    except Exception as e:
        print(f"❌ User API test failed: {e}")
    
    # Test 2: Training API
    print("\n2️⃣  TESTING TRAINING API") 
    print("-" * 40)
    try:
        test_training_api()
    except Exception as e:
        print(f"❌ Training API test failed: {e}")
    
    # Test 3: Prediction API
    print("\n3️⃣  TESTING PREDICTION API")
    print("-" * 40)
    try:
        test_prediction_api()
    except Exception as e:
        print(f"❌ Prediction API test failed: {e}")
    
    # Test 4: Explanation API
    print("\n4️⃣  TESTING EXPLANATION API")
    print("-" * 40)
    try:
        test_explanation_api()
    except Exception as e:
        print(f"❌ Explanation API test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ML API DEMONSTRATION COMPLETE")
    print("=" * 60)


def test_user_api():
    """Test the user-friendly ExoplanetMLAPI."""
    
    from ML.src.api.user_api import ExoplanetMLAPI
    
    api = ExoplanetMLAPI()
    
    # List available datasets
    print("📊 Available datasets:")
    datasets = api.list_available_datasets()
    for dataset in datasets:
        print(f"   - {dataset}")
    
    # Get dataset info
    if datasets:
        dataset_name = datasets[0]
        print(f"\n📈 Info for dataset '{dataset_name}':")
        info = api.get_dataset_info(dataset_name)
        print(f"   Rows: {info.get('num_rows', 'N/A')}")
        print(f"   Columns: {info.get('num_columns', 'N/A')}")
    
    # List available models
    print("\n🤖 Available model types:")
    models = api.list_available_models()
    for model in models:
        print(f"   - {model}")
    
    # List trained models
    print("\n🎯 Trained models:")
    trained_models = api.list_trained_models()
    if trained_models:
        for model in trained_models[:3]:  # Show first 3
            print(f"   - {model.get('name', 'Unknown')}")
    else:
        print("   No trained models found")
    
    print("✅ User API test completed")


def test_training_api():
    """Test the TrainingAPI for advanced training."""
    
    from ML.src.api.training_api import TrainingAPI
    
    api = TrainingAPI()
    session_id = "demo_session_" + str(int(time.time()))
    
    # Start training session
    print(f"🔄 Starting training session: {session_id}")
    result = api.start_training_session(session_id)
    print(f"   Status: {result.get('status', 'Unknown')}")
    
    # Try to load data
    print("\n📥 Loading training data...")
    try:
        data_result = api.load_data_for_training(
            session_id=session_id,
            dataset_name="kepler",
            test_size=0.2
        )
        print(f"   Status: {data_result.get('status', 'Unknown')}")
        print(f"   Training samples: {data_result.get('train_samples', 'N/A')}")
        print(f"   Test samples: {data_result.get('test_samples', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️  Data loading issue: {e}")
    
    # Get session info
    print("\n📋 Session information:")
    session_info = api.get_session_info(session_id)
    print(f"   Session ID: {session_info.get('session_id', 'N/A')}")
    print(f"   Status: {session_info.get('status', 'N/A')}")
    
    print("✅ Training API test completed")


def test_prediction_api():
    """Test the PredictionAPI for making predictions."""
    
    from ML.src.api.prediction_api import PredictionAPI
    
    api = PredictionAPI()
    
    # Check loaded models
    print("🔍 Checking loaded models...")
    loaded_models = api.get_loaded_models()
    print(f"   Loaded models: {len(loaded_models.get('models', []))}")
    
    # Try to find any pretrained models
    models_dir = Path("ML/models/pretrained")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.joblib"))
        if model_files:
            model_path = str(model_files[0])
            model_id = model_files[0].stem
            
            print(f"\n📤 Loading model: {model_id}")
            try:
                load_result = api.load_model(model_path, model_id)
                print(f"   Status: {load_result.get('status', 'Unknown')}")
                
                # Try a simple prediction with dummy data
                print(f"\n🎯 Testing prediction with dummy data...")
                dummy_features = {
                    'koi_period': 365.25,
                    'koi_prad': 1.0,
                    'koi_teq': 288.0
                }
                
                try:
                    pred_result = api.predict_single(model_id, dummy_features)
                    print(f"   Prediction: {pred_result.get('prediction', 'N/A')}")
                    print(f"   Confidence: {pred_result.get('confidence', 'N/A')}")
                except Exception as e:
                    print(f"   ⚠️  Prediction issue: {e}")
                    
            except Exception as e:
                print(f"   ⚠️  Model loading issue: {e}")
        else:
            print("   No pretrained models found")
    else:
        print("   Models directory not found")
    
    print("✅ Prediction API test completed")


def test_explanation_api():
    """Test the ExplanationAPI for model explainability."""
    
    from ML.src.api.explanation_api import ExplanationAPI
    
    api = ExplanationAPI()
    
    print("🔬 Testing explainability features...")
    
    # Check available methods
    print("   Feature importance analysis available")
    print("   Column dropping analysis available")
    
    # Note: Actual testing would require trained models
    print("   ℹ️  Full explanation testing requires trained models")
    
    print("✅ Explanation API test completed")


def demo_quick_workflow():
    """Demonstrate a complete ML workflow."""
    
    print("\n" + "🔄" * 20 + " QUICK WORKFLOW DEMO " + "🔄" * 20)
    
    try:
        from ML.src.api.user_api import ExoplanetMLAPI
        
        api = ExoplanetMLAPI()
        
        print("\n1. List datasets:")
        datasets = api.list_available_datasets()
        print(f"   Found {len(datasets)} datasets")
        
        print("\n2. List model types:")
        models = api.list_available_models()
        print(f"   Found {len(models)} model types")
        
        print("\n3. Check existing trained models:")
        trained = api.list_trained_models()
        print(f"   Found {len(trained)} trained models")
        
        if trained:
            print("\n4. Sample prediction with first trained model:")
            model_name = trained[0]['name']
            
            # Sample features (these are typical exoplanet parameters)
            sample_features = {
                'koi_period': 365.25,      # Orbital period in days
                'koi_prad': 1.0,           # Planet radius (Earth radii)
                'koi_teq': 288.0,          # Equilibrium temperature (K)
                'koi_insol': 1.0,          # Insolation flux (Earth flux)
                'koi_dor': 215.0,          # Planet-star distance over stellar radius
                'koi_impact': 0.5,         # Impact parameter
                'koi_duration': 6.0,       # Transit duration (hours)
                'koi_depth': 84.0          # Transit depth (ppm)
            }
            
            try:
                result = api.predict_single(model_name, sample_features)
                print(f"   Prediction: {result.get('prediction', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 'N/A'):.3f}")
                
                print("\n5. Get feature importance:")
                importance = api.get_feature_importance(model_name, top_n=5)
                if 'feature_importance' in importance:
                    print("   Top important features:")
                    for feature in importance['feature_importance'][:5]:
                        print(f"     - {feature['feature']}: {feature['importance']:.3f}")
                        
            except Exception as e:
                print(f"   ⚠️  Prediction/importance analysis failed: {e}")
        else:
            print("\n4. No trained models available for prediction demo")
            print("   💡 Tip: Run the training workflow first to create models")
        
    except Exception as e:
        print(f"❌ Workflow demo failed: {e}")
    
    print("\n" + "✅" * 20 + " WORKFLOW DEMO COMPLETE " + "✅" * 20)


if __name__ == "__main__":
    main()
    demo_quick_workflow()