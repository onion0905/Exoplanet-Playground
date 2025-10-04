"""
User-friendly API for NASA Exoplanet ML System.
Provides simple interfaces for training, prediction, and feature analysis.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ML.src.data.data_loader import DataLoader
from ML.src.data.data_validator import DataValidator
from ML.src.data.data_processor import DataProcessor
from ML.src.utils.model_factory import ModelFactory


class ExoplanetMLAPI:
    """
    User-friendly API for the NASA Exoplanet ML System.
    
    This API provides simple methods for:
    - Training models on NASA datasets
    - Making predictions on new data
    - Analyzing feature importance
    - Managing pretrained models
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.data_processor = DataProcessor()
        self.model_factory = ModelFactory()
        
        # Ensure directories exist - use ML project directory
        ml_project_root = Path(__file__).parent.parent.parent
        self.models_dir = ml_project_root / "models"
        self.pretrained_dir = self.models_dir / "pretrained"
        self.user_dir = self.models_dir / "user"
        
        for dir_path in [self.models_dir, self.pretrained_dir, self.user_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # DATASET MANAGEMENT
    # ========================================
    
    def list_available_datasets(self) -> List[str]:
        """Get list of available NASA datasets."""
        return ['kepler', 'tess', 'k2']
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        try:
            raw_data = self.data_loader.load_nasa_dataset(dataset_name)
            clean_data, report = self.data_validator.create_clean_dataset(raw_data)
            
            return {
                'dataset_name': dataset_name,
                'total_records': len(raw_data),
                'clean_records': len(clean_data),
                'original_features': len(raw_data.columns),
                'clean_features': len(clean_data.columns),
                'removed_features': len(raw_data.columns) - len(clean_data.columns),
                'dataset_type': report['dataset_type'],
                'target_column': report.get('target_column', 'Unknown')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_sample_data(self, dataset_name: str, n_samples: int = 5) -> Dict[str, Any]:
        """Get sample data from a dataset for inspection."""
        try:
            raw_data = self.data_loader.load_nasa_dataset(dataset_name)
            clean_data, _ = self.data_validator.create_clean_dataset(raw_data)
            
            sample = clean_data.head(n_samples)
            
            return {
                'dataset_name': dataset_name,
                'sample_size': len(sample),
                'features': list(sample.columns),
                'sample_data': sample.to_dict('records')
            }
        except Exception as e:
            return {'error': str(e)}
    
    # ========================================
    # MODEL TRAINING
    # ========================================
    
    def list_available_models(self) -> List[str]:
        """Get list of available model types."""
        return ['random_forest', 'decision_tree', 'linear_regression', 'svm', 'xgboost', 'pca', 'deep_learning']
    
    def train_model(self, model_type: str, dataset_name: str, 
                   model_name: Optional[str] = None,
                   hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a model on a specified dataset.
        
        Args:
            model_type: Type of model ('random_forest', 'decision_tree', etc.)
            dataset_name: Dataset to train on ('kepler', 'tess', 'k2')
            model_name: Optional custom name for the model
            hyperparameters: Optional hyperparameters for the model
            
        Returns:
            Dictionary with training results and model information
        """
        try:
            start_time = time.time()
            
            # Load and prepare data
            raw_data = self.data_loader.load_nasa_dataset(dataset_name)
            clean_data, report = self.data_validator.create_clean_dataset(raw_data)
            
            # Get target column for dataset
            target_columns = {
                'kepler': 'koi_disposition',
                'tess': 'tfopwg_disp',
                'k2': 'disposition'
            }
            target_column = target_columns.get(dataset_name, 'koi_disposition')
            
            # Prepare data with train/val/test splits
            processed_data = self.data_processor.prepare_data(
                clean_data, target_column, test_size=0.2, val_size=0.2
            )
            
            # Create model
            model = self.model_factory.create_model(model_type)
            
            # Build model with hyperparameters
            if hyperparameters:
                model.build_model(**hyperparameters)
            else:
                model.build_model()
            
            # Train model
            training_result = model.train(
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_val'],
                processed_data['y_val']
            )
            
            # Test model performance
            test_predictions = model.predict(processed_data['X_test'])
            test_accuracy = (test_predictions == processed_data['y_test']).mean()
            
            # Save model
            if model_name is None:
                model_name = f"{model_type}_{dataset_name}_{int(time.time())}"
            
            model_path = self.user_dir / f"{model_name}.joblib"
            
            model.save_model(str(model_path))
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': model_type,
                'dataset_name': dataset_name,
                'training_time': time.time() - start_time,
                'train_accuracy': training_result.get('train_accuracy', 'N/A'),
                'val_accuracy': training_result.get('val_accuracy', 'N/A'),
                'test_accuracy': float(test_accuracy),
                'feature_count': len(processed_data['feature_names']),
                'train_size': len(processed_data['X_train']),
                'val_size': len(processed_data['X_val']),
                'test_size': len(processed_data['X_test']),
                'hyperparameters': hyperparameters or {},
                'feature_names': processed_data['feature_names'],
                'model_path': str(model_path),
                'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_path = self.user_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return {
                'success': True,
                'model_name': model_name,
                'model_type': model_type,
                'dataset': dataset_name,
                'training_time': metadata['training_time'],
                'test_accuracy': test_accuracy,
                'feature_count': metadata['feature_count'],
                'model_path': str(model_path),
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_type': model_type,
                'dataset': dataset_name
            }
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained models with their metadata."""
        models = []
        
        for metadata_file in self.user_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                models.append(metadata)
            except Exception as e:
                continue
                
        return models
    
    # ========================================
    # PREDICTIONS
    # ========================================
    
    def predict_single(self, model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single exoplanet candidate.
        
        Args:
            model_name: Name of the trained model to use
            features: Dictionary of feature values
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load model metadata
            metadata_path = self.user_dir / f"{model_name}_metadata.json"
            if not metadata_path.exists():
                return {'success': False, 'error': f'Model {model_name} not found'}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load model
            model = self.model_factory.create_model(metadata['model_type'])
            model.load_model(metadata['model_path'])
            
            # Prepare input data
            feature_names = metadata['feature_names']
            
            # Create DataFrame with proper features
            input_data = pd.DataFrame([features])
            
            # Ensure all required features are present
            missing_features = set(feature_names) - set(input_data.columns)
            if missing_features:
                return {
                    'success': False,
                    'error': f'Missing required features: {list(missing_features)}'
                }
            
            # Select and order features correctly
            input_data = input_data[feature_names]
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Get probability if available
            probability = None
            confidence = 'Unknown'
            
            if hasattr(model.model, 'predict_proba'):
                prob_array = model.model.predict_proba(input_data)[0]
                probability = float(max(prob_array))
                
                if probability > 0.8:
                    confidence = 'High'
                elif probability > 0.6:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
            
            return {
                'success': True,
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'model_used': model_name,
                'model_type': metadata['model_type'],
                'dataset_trained_on': metadata['dataset_name']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def predict_batch(self, model_name: str, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple exoplanet candidates."""
        results = []
        
        for i, features in enumerate(features_list):
            result = self.predict_single(model_name, features)
            result['sample_id'] = i
            results.append(result)
        
        return results
    
    # ========================================
    # FEATURE ANALYSIS  
    # ========================================
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Get feature importance for a trained model.
        
        Args:
            model_name: Name of the trained model
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance information
        """
        try:
            # Load model metadata
            metadata_path = self.user_dir / f"{model_name}_metadata.json"
            if not metadata_path.exists():
                return {'success': False, 'error': f'Model {model_name} not found'}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load model
            model = self.model_factory.create_model(metadata['model_type'])
            model.load_model(metadata['model_path'])
            
            # Get feature importance if available
            if hasattr(model.model, 'feature_importances_'):
                feature_names = metadata['feature_names']
                importances = model.model.feature_importances_
                
                # Create feature importance dictionary
                feature_importance = dict(zip(feature_names, importances))
                
                # Sort by importance
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                return {
                    'success': True,
                    'model_name': model_name,
                    'model_type': metadata['model_type'],
                    'top_features': dict(sorted_features[:top_n]),
                    'all_features': feature_importance,
                    'total_features': len(feature_names)
                }
            
            elif hasattr(model.model, 'coef_'):
                # For linear models, use coefficient magnitudes
                feature_names = metadata['feature_names']
                coefficients = model.model.coef_[0] if len(model.model.coef_.shape) > 1 else model.model.coef_
                
                # Use absolute values for importance
                importances = np.abs(coefficients)
                feature_importance = dict(zip(feature_names, importances))
                
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                return {
                    'success': True,
                    'model_name': model_name,
                    'model_type': metadata['model_type'],
                    'top_features': dict(sorted_features[:top_n]),
                    'all_features': feature_importance,
                    'total_features': len(feature_names),
                    'note': 'Importance based on coefficient magnitudes'
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Model type {metadata["model_type"]} does not support feature importance'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def analyze_prediction_factors(self, model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze which features contributed most to a specific prediction.
        
        Args:
            model_name: Name of the trained model
            features: Feature values for the prediction
            
        Returns:
            Dictionary with prediction analysis
        """
        try:
            # First make the prediction
            prediction_result = self.predict_single(model_name, features)
            if not prediction_result['success']:
                return prediction_result
            
            # Get feature importance
            importance_result = self.get_feature_importance(model_name, top_n=5)
            if not importance_result['success']:
                return importance_result
            
            # Combine results
            return {
                'success': True,
                'prediction': prediction_result['prediction'],
                'probability': prediction_result['probability'],
                'top_contributing_features': importance_result['top_features'],
                'feature_values': {k: v for k, v in features.items() 
                                 if k in importance_result['top_features']},
                'model_name': model_name
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # ========================================
    # PRETRAINED MODELS
    # ========================================
    
    def generate_pretrained_models(self) -> Dict[str, Any]:
        """Generate a set of pretrained models for immediate use."""
        results = {}
        
        # Define models to train
        models_config = [
            ('random_forest', 'kepler'),
            ('decision_tree', 'kepler'),
            ('linear_regression', 'kepler'),
            ('random_forest', 'tess'),
            ('decision_tree', 'tess')
        ]
        
        print("🔄 Generating pretrained models...")
        
        for model_type, dataset in models_config:
            print(f"   Training {model_type} on {dataset}...")
            
            result = self.train_model(
                model_type=model_type,
                dataset_name=dataset,
                model_name=f"pretrained_{model_type}_{dataset}"
            )
            
            results[f"{model_type}_{dataset}"] = result
            
            if result['success']:
                print(f"   ✅ {model_type} on {dataset}: {result['test_accuracy']:.3f} accuracy")
            else:
                print(f"   ❌ {model_type} on {dataset}: {result['error']}")
        
        return results


# ========================================
# CONVENIENCE FUNCTIONS FOR USERS
# ========================================

def quick_start_example():
    """Provide a quick start example for users."""
    api = ExoplanetMLAPI()
    
    print("🚀 NASA Exoplanet ML API - Quick Start Example")
    print("=" * 50)
    
    # Show available datasets
    datasets = api.list_available_datasets()
    print(f"📊 Available datasets: {datasets}")
    
    # Show dataset info
    kepler_info = api.get_dataset_info('kepler')
    print(f"📈 Kepler dataset: {kepler_info['clean_records']} records, {kepler_info['clean_features']} features")
    
    # Train a simple model
    print("\n🔄 Training a Random Forest model...")
    train_result = api.train_model('random_forest', 'kepler', 'example_model')
    
    if train_result['success']:
        print(f"✅ Model trained! Accuracy: {train_result['test_accuracy']:.3f}")
        
        # Get sample data for prediction
        sample = api.get_sample_data('kepler', 1)
        if sample and not 'error' in sample:
            features = sample['sample_data'][0]
            
            # Remove target column if present
            target_cols = ['koi_disposition', 'tfopwg_disp', 'disposition']
            for col in target_cols:
                features.pop(col, None)
            
            print(f"\n🔮 Making a prediction...")
            prediction = api.predict_single('example_model', features)
            
            if prediction['success']:
                print(f"✅ Prediction: {prediction['prediction']} (confidence: {prediction['confidence']})")
                
                # Show feature importance
                importance = api.get_feature_importance('example_model', 3)
                if importance['success']:
                    print(f"📊 Top 3 important features:")
                    for feature, score in importance['top_features'].items():
                        print(f"   - {feature}: {score:.3f}")
            else:
                print(f"❌ Prediction failed: {prediction['error']}")
        else:
            print("❌ Could not get sample data")
    else:
        print(f"❌ Training failed: {train_result['error']}")


# Alias for backwards compatibility
UserAPI = ExoplanetMLAPI

if __name__ == "__main__":
    quick_start_example()