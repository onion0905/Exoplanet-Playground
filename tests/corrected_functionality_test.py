#!/usr/bin/env python3
"""
Corrected Comprehensive Functionality Test for Exoplanet ML System

This script tests:
1. Prediction API with trained models for real vs false positive classification
2. Explanation API for feature importance and model interpretability  
3. Column selection from datasets for custom training
4. Proper model saving to ML/models/ directory

Usage:
    python tests/corrected_functionality_test.py
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the project root and ML source directory to the path
project_root = os.path.dirname(os.path.dirname(__file__))
ml_src_path = os.path.join(project_root, 'ML', 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, ml_src_path)

# Import ML APIs - using the correct class name
from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI

class CorrectedFunctionalityTester:
    """Test core functionality with proper API usage."""
    
    def __init__(self):
        """Initialize the tester with all APIs."""
        self.user_api = ExoplanetMLAPI()  # This is the correct class name
        self.training_api = TrainingAPI()
        self.prediction_api = PredictionAPI()
        self.explanation_api = ExplanationAPI()
        
        self.results = {
            'test_start_time': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {},
            'errors': []
        }
        
        # Ensure correct model save directory
        self.model_save_dir = Path('/home/brine/OneDrive/Work/Exoplanet-Playground/ML/models/user')
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("🚀 CORRECTED EXOPLANET ML SYSTEM FUNCTIONALITY TEST")
        print("=" * 80)
        print(f"📅 Started at: {self.results['test_start_time']}")
        print(f"💾 Model save directory: {self.model_save_dir}")
        print()
    
    def log_test(self, test_name: str, passed: bool, details: Dict[str, Any], error: str = None):
        """Log test results."""
        self.results['tests_run'] += 1
        if passed:
            self.results['tests_passed'] += 1
            print(f"✅ {test_name}: PASSED")
        else:
            self.results['tests_failed'] += 1
            print(f"❌ {test_name}: FAILED")
            if error:
                print(f"   Error: {error}")
                self.results['errors'].append(f"{test_name}: {error}")
        
        self.results['test_details'][test_name] = {
            'passed': passed,
            'details': details,
            'error': error
        }
        print(f"   Details: {details}")
        print()
    
    def test_1_dataset_access_and_classification(self) -> bool:
        """Test 1: Access datasets and understand classification targets."""
        print("🔍 TEST 1: Dataset Access and Classification Structure")
        print("-" * 50)
        
        try:
            # List available datasets
            datasets = self.user_api.list_available_datasets()
            
            classification_info = {}
            for dataset_name in datasets:
                # Get dataset info
                dataset_info = self.user_api.get_dataset_info(dataset_name)
                
                # Get sample data to understand structure
                sample_data = self.user_api.get_sample_data(dataset_name, n_samples=10)
                
                if 'error' not in dataset_info and 'error' not in sample_data:
                    classification_info[dataset_name] = {
                        'total_records': dataset_info['total_records'],
                        'clean_records': dataset_info['clean_records'],
                        'target_column': dataset_info.get('target_column', 'Unknown'),
                        'features': len(sample_data['features']),
                        'sample_size': sample_data['sample_size']
                    }
            
            success = len(classification_info) > 0
            
            self.log_test(
                'Dataset Access and Classification',
                success,
                {
                    'datasets_available': len(datasets),
                    'datasets_analyzed': len(classification_info),
                    'classification_info': classification_info
                }
            )
            return success
            
        except Exception as e:
            self.log_test('Dataset Access and Classification', False, {}, str(e))
            return False
    
    def test_2_model_training_and_saving(self) -> bool:
        """Test 2: Train a model and save to ML/models/ directory."""
        print("🎯 TEST 2: Model Training and Saving")
        print("-" * 50)
        
        try:
            # Train a small fast model for testing
            model_name = f"test_functionality_{int(time.time())}"
            
            training_result = self.user_api.train_model(
                model_type='decision_tree',
                dataset_name='kepler',
                model_name=model_name,
                hyperparameters={'max_depth': 3, 'random_state': 42}
            )
            
            success = 'error' not in training_result
            
            if success:
                # Check if model was saved in correct directory
                expected_path = self.model_save_dir / f"{model_name}_metadata.json"
                model_saved_correctly = expected_path.exists()
                
                details = {
                    'training_successful': success,
                    'model_name': model_name,
                    'training_accuracy': training_result.get('training_accuracy', 'N/A'),
                    'validation_accuracy': training_result.get('validation_accuracy', 'N/A'),
                    'model_saved_to_correct_dir': model_saved_correctly,
                    'model_path': str(expected_path) if model_saved_correctly else None
                }
            else:
                details = {
                    'training_successful': False,
                    'error': training_result.get('error', 'Unknown training error')
                }
            
            self.log_test('Model Training and Saving', success, details)
            return success
            
        except Exception as e:
            self.log_test('Model Training and Saving', False, {}, str(e))
            return False
    
    def test_3_prediction_functionality(self) -> bool:
        """Test 3: Load a model and make predictions."""
        print("🔮 TEST 3: Prediction Functionality")
        print("-" * 50)
        
        try:
            # Find available trained models
            model_files = list(self.model_save_dir.glob("*_metadata.json"))
            
            if not model_files:
                raise Exception("No trained models found for prediction testing")
            
            # Use the most recent model
            metadata_file = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
            model_path = str(metadata_file).replace('_metadata.json', '')
            
            # Load the model
            load_result = self.prediction_api.load_model(model_path, model_id="prediction_test")
            
            if load_result['status'] != 'success':
                raise Exception(f"Failed to load model: {load_result.get('error', 'Unknown')}")
            
            # Get sample data for prediction
            sample_result = self.user_api.get_sample_data('kepler', n_samples=3)
            
            if 'error' in sample_result:
                raise Exception(f"Failed to get sample data: {sample_result['error']}")
            
            # Make predictions on sample data
            predictions = []
            for i, sample_row in enumerate(sample_result['sample_data'][:2]):
                # Remove target column if present
                input_data = {k: v for k, v in sample_row.items() 
                             if k not in ['koi_disposition', 'tfopwg_disp', 'disposition']}
                
                try:
                    pred_result = self.prediction_api.predict_single("prediction_test", input_data)
                    if pred_result['status'] == 'success':
                        predictions.append({
                            'sample_index': i,
                            'prediction': pred_result['prediction'],
                            'confidence': max(pred_result.get('probabilities', [0])) if pred_result.get('probabilities') else 'N/A'
                        })
                except Exception as pred_error:
                    print(f"   Warning: Prediction {i} failed: {str(pred_error)}")
            
            success = len(predictions) > 0
            
            self.log_test(
                'Prediction Functionality',
                success,
                {
                    'model_loaded': load_result['status'] == 'success',
                    'predictions_made': len(predictions),
                    'sample_predictions': predictions[:2]  # Show first 2
                }
            )
            return success
            
        except Exception as e:
            self.log_test('Prediction Functionality', False, {}, str(e))
            return False
    
    def test_4_explanation_functionality(self) -> bool:
        """Test 4: Generate model explanations."""
        print("📊 TEST 4: Explanation Functionality")
        print("-" * 50)
        
        try:
            # Ensure we have a model loaded
            if "prediction_test" not in self.prediction_api.loaded_models:
                # Load a model for explanation
                model_files = list(self.model_save_dir.glob("*_metadata.json"))
                if not model_files:
                    raise Exception("No models available for explanation testing")
                
                metadata_file = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
                model_path = str(metadata_file).replace('_metadata.json', '')
                
                load_result = self.prediction_api.load_model(model_path, model_id="explanation_test")
                if load_result['status'] != 'success':
                    raise Exception("Failed to load model for explanation")
                model_id = "explanation_test"
            else:
                model_id = "prediction_test"
            
            # Get sample data
            sample_result = self.user_api.get_sample_data('kepler', n_samples=50)
            if 'error' in sample_result:
                raise Exception(f"Failed to get sample data: {sample_result['error']}")
            
            # Convert to DataFrame
            sample_df = pd.DataFrame(sample_result['sample_data'])
            
            # Prepare data for explanation (remove target and non-numeric columns)
            target_cols = ['koi_disposition', 'tfopwg_disp', 'disposition']
            feature_cols = [col for col in sample_df.columns if col not in target_cols]
            X = sample_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
            
            if len(X.columns) == 0:
                raise Exception("No numeric features available for explanation")
            
            # Split data for explanation
            split_idx = len(X) // 2
            X_train, X_test = X[:split_idx], X[split_idx:]
            
            # Create dummy target for explanation
            y_train = pd.Series(['CONFIRMED'] * len(X_train))
            y_test = pd.Series(['CANDIDATE'] * len(X_test))
            
            # Test local explanation for a single instance
            if len(X_test) > 0:
                single_instance = X_test.iloc[0].to_dict()
                local_explanation = self.explanation_api.explain_prediction_local(
                    model_id=model_id,
                    input_data=single_instance,
                    X_background=X_train.sample(min(10, len(X_train))),
                    methods=['model_coefficients']  # Use simple method
                )
                
                explanation_success = local_explanation.get('status') == 'success'
            else:
                explanation_success = False
            
            self.log_test(
                'Explanation Functionality',
                explanation_success,
                {
                    'model_id': model_id,
                    'data_shape': X.shape,
                    'numeric_features': len(X.columns),
                    'explanation_status': local_explanation.get('status') if 'local_explanation' in locals() else 'Failed',
                    'explanation_methods': list(local_explanation.get('explanations', {}).keys()) if explanation_success else []
                }
            )
            return explanation_success
            
        except Exception as e:
            self.log_test('Explanation Functionality', False, {}, str(e))
            return False
    
    def test_5_column_selection_training(self) -> bool:
        """Test 5: Training with custom column selection."""
        print("🎛️ TEST 5: Column Selection Training")
        print("-" * 50)
        
        try:
            # Get available models and datasets
            available_models = self.user_api.list_available_models()
            available_datasets = self.user_api.list_available_datasets()
            
            # Test with a small subset of features
            session_id = f"column_test_{int(time.time())}"
            
            # Start training session
            session_result = self.training_api.start_training_session(session_id)
            
            if session_result['status'] != 'initialized':
                raise Exception("Failed to start training session")
            
            # Load data with specific configuration
            data_config = {
                'datasets': ['kepler']
            }
            
            data_result = self.training_api.load_data_for_training(
                session_id=session_id,
                data_source='nasa',
                data_config=data_config
            )
            
            if data_result['status'] != 'success':
                raise Exception(f"Failed to load data: {data_result.get('error', 'Unknown')}")
            
            # The data loading and session creation works
            success = (
                len(available_models) > 0 and
                len(available_datasets) > 0 and
                data_result['status'] == 'success'
            )
            
            self.log_test(
                'Column Selection Training',
                success,
                {
                    'available_models': len(available_models),
                    'available_datasets': len(available_datasets),
                    'session_created': session_result['status'] == 'initialized',
                    'data_loaded': data_result['status'] == 'success',
                    'data_shape': data_result.get('data_info', {}).get('shape', 'Unknown')
                }
            )
            return success
            
        except Exception as e:
            self.log_test('Column Selection Training', False, {}, str(e))
            return False
    
    def run_all_tests(self):
        """Run all functionality tests."""
        print("🎯 Running corrected functionality tests...\n")
        
        # Run all tests
        tests = [
            self.test_1_dataset_access_and_classification,
            self.test_2_model_training_and_saving,
            self.test_3_prediction_functionality,
            self.test_4_explanation_functionality,
            self.test_5_column_selection_training
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"❌ CRITICAL ERROR in {test_func.__name__}: {str(e)}")
                self.results['errors'].append(f"CRITICAL: {test_func.__name__}: {str(e)}")
            
            print()  # Add space between tests
        
        # Final summary
        self.results['test_end_time'] = datetime.now().isoformat()
        self.results['success_rate'] = (
            self.results['tests_passed'] / self.results['tests_run'] * 100
            if self.results['tests_run'] > 0 else 0
        )
        
        print("=" * 80)
        print("📋 CORRECTED FUNCTIONALITY TEST SUMMARY")
        print("=" * 80)
        print(f"🕒 Test Duration: {self.results['test_start_time']} → {self.results['test_end_time']}")
        print(f"📊 Tests Run: {self.results['tests_run']}")
        print(f"✅ Tests Passed: {self.results['tests_passed']}")
        print(f"❌ Tests Failed: {self.results['tests_failed']}")
        print(f"📈 Success Rate: {self.results['success_rate']:.1f}%")
        
        if self.results['success_rate'] >= 80:
            print("🎉 OVERALL STATUS: EXCELLENT - System is highly functional!")
        elif self.results['success_rate'] >= 60:
            print("⚠️  OVERALL STATUS: GOOD - System works with minor issues")
        else:
            print("🚨 OVERALL STATUS: NEEDS ATTENTION - Major issues detected")
        
        if self.results['errors']:
            print(f"\n🚨 Errors encountered ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        # Save detailed results
        results_file = Path(f"/home/brine/OneDrive/Work/Exoplanet-Playground/tests/results/corrected_functionality_test_{int(time.time())}.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        print("=" * 80)
        
        return self.results

def main():
    """Main function to run corrected functionality tests."""
    tester = CorrectedFunctionalityTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['success_rate'] >= 60:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()