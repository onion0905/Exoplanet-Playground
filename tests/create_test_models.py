#!/usr/bin/env python3
"""
Create Working Models for Testing

This script creates functional models that can be loaded and used for predictions.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
ml_src_path = os.path.join(project_root, 'ML', 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, ml_src_path)

from ML.src.api.user_api import ExoplanetMLAPI

def create_test_models():
    """Create a few small, fast models for testing."""
    
    print("🔧 CREATING TEST MODELS FOR VALIDATION")
    print("=" * 50)
    
    # Initialize API
    user_api = ExoplanetMLAPI()
    
    # Model configurations (small and fast)
    models_to_create = [
        {
            'model_type': 'decision_tree',
            'dataset_name': 'kepler',
            'model_name': 'test_dt_kepler',
            'hyperparameters': {'max_depth': 5, 'random_state': 42}
        },
        {
            'model_type': 'random_forest', 
            'dataset_name': 'tess',
            'model_name': 'test_rf_tess',
            'hyperparameters': {'n_estimators': 10, 'max_depth': 3, 'random_state': 42}
        },
        {
            'model_type': 'linear_regression',
            'dataset_name': 'k2',
            'model_name': 'test_lr_k2',
            'hyperparameters': {'random_state': 42}
        }
    ]
    
    created_models = []
    
    for config in models_to_create:
        print(f"\n🔨 Creating {config['model_type']} model on {config['dataset_name']}...")
        
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
                created_models.append({
                    'model_name': config['model_name'],
                    'model_type': config['model_type'],
                    'dataset': config['dataset_name'],
                    'accuracy': result.get('test_accuracy', 'N/A'),
                    'training_time': duration
                })
                print(f"   ✅ Success! Accuracy: {result.get('test_accuracy', 'N/A'):.3f}, Time: {duration:.1f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print(f"\n📦 Successfully created {len(created_models)} models:")
    for model in created_models:
        print(f"   • {model['model_name']}: {model['model_type']} on {model['dataset']} ({model['accuracy']:.3f} accuracy)")
    
    return created_models

if __name__ == "__main__":
    models = create_test_models()
    print(f"\n🎉 Created {len(models)} test models ready for validation!")