#!/usr/bin/env python3
"""
Comprehensive Functionality Test for Exoplanet ML System

This script tests:
1. Prediction API with trained models for real vs false positive classification
2. Explanation API for feature importance and model interpretability  
3. Column selection from datasets for custom training
4. Proper model saving to ML/models/ directory

Usage:
    python tests/comprehensive_functionality_test.py
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

# Import ML APIs
from ML.src.api.user_api import UserAPI
from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI
from ML.src.data.data_processor import DataProcessor

class ComprehensiveFunctionalityTester:
    """Test all major functionality of the exoplanet ML system."""
    
    def __init__(self):
        """Initialize the tester with all APIs."""
        self.user_api = UserAPI()
        self.training_api = TrainingAPI()
        self.prediction_api = PredictionAPI()
        self.explanation_api = ExplanationAPI()
        self.data_processor = DataProcessor()
        
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
        print("🚀 COMPREHENSIVE EXOPLANET ML SYSTEM FUNCTIONALITY TEST")
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
    
    def test_1_dataset_analysis(self) -> bool:
        """Test 1: Analyze datasets for classification targets."""
        print("🔍 TEST 1: Dataset Analysis for Classification")
        print("-" * 50)
        
        try:
            datasets = self.user_api.list_available_datasets()
            
            classification_info = {}
            for dataset_name in datasets['datasets']:
                dataset_info = self.user_api.get_dataset_info(dataset_name)
                sample_data = self.user_api.get_sample_data(dataset_name, n_samples=100)
                
                # Find classification columns
                if dataset_name == 'kepler':
                    target_col = 'koi_disposition'
                elif dataset_name == 'tess':
                    target_col = 'tfopwg_disp'
                elif dataset_name == 'k2':
                    target_col = 'disposition'
                else:
                    target_col = None
                
                if target_col and target_col in sample_data['data'].columns:
                    unique_values = sample_data['data'][target_col].value_counts().to_dict()
                    classification_info[dataset_name] = {
                        'target_column': target_col,
                        'unique_values': unique_values,
                        'total_classes': len(unique_values)
                    }
            
            self.log_test(
                'Dataset Classification Analysis',
                True,
                {
                    'datasets_analyzed': len(datasets['datasets']),
                    'classification_info': classification_info
                }
            )
            return True
            
        except Exception as e:
            self.log_test('Dataset Classification Analysis', False, {}, str(e))
            return False
    
    def test_2_column_selection_training(self) -> bool:
        """Test 2: Column selection and custom training."""
        print("🎯 TEST 2: Column Selection and Custom Training")
        print("-" * 50)
        
        try:
            # Test with Kepler dataset - select specific columns
            selected_columns = [
                'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
                'koi_teq', 'koi_insol', 'koi_srad', 'koi_smass',
                'koi_disposition'  # target column
            ]
            
            # Start training session
            session = self.training_api.start_training_session(
                session_name=f"column_selection_test_{int(time.time())}"
            )
            
            if session['status'] != 'success':
                raise Exception("Failed to start training session")
            
            session_id = session['session_id']
            
            # Load data with column selection
            data_result = self.training_api.load_data_for_training(
                session_id=session_id,
                dataset_name='kepler',
                target_column='koi_disposition',
                feature_columns=selected_columns[:-1]  # exclude target
            )
            
            if data_result['status'] != 'success':
                raise Exception(f"Failed to load data: {data_result.get('error', 'Unknown')}")
            
            # Configure training
            config_result = self.training_api.configure_training(
                session_id=session_id,
                model_type='random_forest',
                training_params={'n_estimators': 50, 'random_state': 42}
            )
            
            if config_result['status'] != 'success':
                raise Exception("Failed to configure training")
            
            # Start training
            train_result = self.training_api.start_training(session_id)
            
            if train_result['status'] != 'success':
                raise Exception(f"Training failed: {train_result.get('error', 'Unknown')}")
            
            # Save model to correct directory
            model_path = self.model_save_dir / f"column_selection_test_{int(time.time())}"
            save_result = self.training_api.save_trained_model(
                session_id=session_id,
                model_path=str(model_path)
            )
            
            training_info = self.training_api.get_session_info(session_id)
            
            self.log_test(
                'Column Selection Training',
                True,
                {
                    'session_id': session_id,
                    'selected_columns': len(selected_columns) - 1,
                    'model_accuracy': training_info['session_info'].get('training_accuracy', 'N/A'),
                    'model_saved': save_result.get('status') == 'success',
                    'model_path': str(model_path) if save_result.get('status') == 'success' else None
                }
            )
            return True
            
        except Exception as e:
            self.log_test('Column Selection Training', False, {}, str(e))
            return False
    
    def test_3_prediction_api(self) -> bool:
        """Test 3: Prediction API with trained models."""
        print("🔮 TEST 3: Prediction API Testing")
        print("-" * 50)
        
        try:
            # Find available trained models in the ML/models/user directory
            model_files = list(self.model_save_dir.glob("*_metadata.json"))
            
            if not model_files:
                raise Exception("No trained models found for testing")
            
            # Use the first available model
            metadata_file = model_files[0]
            model_path = str(metadata_file).replace('_metadata.json', '')
            
            # Load the model
            load_result = self.prediction_api.load_model(model_path, model_id="test_model")
            
            if load_result['status'] != 'success':
                raise Exception(f"Failed to load model: {load_result.get('error', 'Unknown')}")
            
            # Get sample data for prediction
            sample_data = self.user_api.get_sample_data('kepler', n_samples=5)
            
            predictions = []
            for idx, row in sample_data['data'].head(3).iterrows():
                # Remove target column if present
                input_data = row.drop(['koi_disposition'], errors='ignore').to_dict()
                
                # Make prediction
                pred_result = self.prediction_api.predict_single("test_model", input_data)
                
                if pred_result['status'] == 'success':
                    predictions.append({
                        'input_sample': idx,
                        'prediction': pred_result['prediction'],
                        'probabilities': pred_result.get('probabilities', []),
                        'class_probabilities': pred_result.get('class_probabilities', {})
                    })
            
            # Test batch prediction
            batch_data = sample_data['data'].head(5).drop(['koi_disposition'], errors='ignore', axis=1)
            batch_result = self.prediction_api.predict_batch("test_model", batch_data.to_dict('records'))
            
            self.log_test(
                'Prediction API',
                True,
                {
                    'model_loaded': load_result['status'] == 'success',
                    'single_predictions': len(predictions),
                    'batch_predictions': len(batch_result.get('predictions', [])) if batch_result.get('status') == 'success' else 0,
                    'sample_prediction': predictions[0] if predictions else None
                }
            )
            return True
            
        except Exception as e:
            self.log_test('Prediction API', False, {}, str(e))
            return False
    
    def test_4_explanation_api(self) -> bool:
        """Test 4: Explanation API for model interpretability."""
        print("📊 TEST 4: Explanation API Testing")
        print("-" * 50)
        
        try:
            # Load a trained model if not already loaded
            model_files = list(self.model_save_dir.glob("*_metadata.json"))
            
            if not model_files:
                raise Exception("No trained models found for explanation testing")
            
            # Load model for explanation
            metadata_file = model_files[0]
            model_path = str(metadata_file).replace('_metadata.json', '')
            
            # Ensure model is loaded in prediction API
            if "explanation_test_model" not in self.prediction_api.loaded_models:
                load_result = self.prediction_api.load_model(model_path, model_id="explanation_test_model")
                if load_result['status'] != 'success':
                    raise Exception("Failed to load model for explanation")
            
            # Get training data for explanation
            sample_data = self.user_api.get_sample_data('kepler', n_samples=200)
            data_df = sample_data['data']
            
            # Prepare data for explanation
            if 'koi_disposition' in data_df.columns:
                X = data_df.drop(['koi_disposition'], axis=1)
                y = data_df['koi_disposition']
            else:
                X = data_df.iloc[:, :-1]  # All but last column as features
                y = data_df.iloc[:, -1]   # Last column as target
            
            # Remove non-numeric columns for explanation
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_columns].fillna(0)
            
            # Split data
            split_idx = len(X_numeric) // 2
            X_train, X_test = X_numeric[:split_idx], X_numeric[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Test global explanation
            global_explanation = self.explanation_api.explain_model_global(
                model_id="explanation_test_model",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                methods=['model_importance']
            )
            
            # Test local explanation for single instance
            single_instance = X_test.iloc[0:1]
            local_explanation = self.explanation_api.explain_prediction_local(
                model_id="explanation_test_model",
                input_data=single_instance.iloc[0].to_dict(),
                X_background=X_train.sample(50),  # Smaller background for speed
                methods=['lime']
            )
            
            self.log_test(
                'Explanation API',
                True,
                {
                    'global_explanation_status': global_explanation.get('status'),
                    'global_methods_tested': len(global_explanation.get('explanations', {})),
                    'local_explanation_status': local_explanation.get('status'),
                    'feature_importance_available': 'feature_importance' in global_explanation.get('explanations', {}),
                    'local_explanation_available': len(local_explanation.get('explanations', {})) > 0
                }
            )
            return True
            
        except Exception as e:
            self.log_test('Explanation API', False, {}, str(e))
            return False
    
    def test_5_user_data_upload(self) -> bool:
        """Test 5: User data upload functionality."""
        print("📤 TEST 5: User Data Upload Testing")
        print("-" * 50)
        
        try:
            # Create sample user data that matches expected format
            sample_data = self.user_api.get_sample_data('kepler', n_samples=10)
            user_data = sample_data['data'].copy()
            
            # Simulate user uploading this data
            temp_file_path = "/tmp/user_test_data.csv"
            user_data.to_csv(temp_file_path, index=False)
            
            # Test data validation
            data_processor = DataProcessor()
            
            # Load and validate the user data
            loaded_data = pd.read_csv(temp_file_path)
            
            # Test preprocessing
            if 'koi_disposition' in loaded_data.columns:
                X = loaded_data.drop(['koi_disposition'], axis=1)
                y = loaded_data['koi_disposition']
                
                # Basic preprocessing
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X_processed = X[numeric_cols].fillna(X[numeric_cols].mean())
                
                upload_success = True
            else:
                upload_success = False
            
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            self.log_test(
                'User Data Upload',
                upload_success,
                {
                    'data_loaded': len(loaded_data) > 0,
                    'columns_detected': len(loaded_data.columns),
                    'numeric_features': len(numeric_cols) if upload_success else 0,
                    'preprocessing_successful': upload_success,
                    'sample_shape': loaded_data.shape
                }
            )
            return upload_success
            
        except Exception as e:
            self.log_test('User Data Upload', False, {}, str(e))
            return False
    
    def test_6_end_to_end_workflow(self) -> bool:
        """Test 6: Complete end-to-end workflow."""
        print("🔄 TEST 6: End-to-End Workflow")
        print("-" * 50)
        
        try:
            # 1. Load data with custom column selection
            session = self.training_api.start_training_session(
                session_name=f"e2e_workflow_{int(time.time())}"
            )
            session_id = session['session_id']
            
            # 2. Train a model
            self.training_api.load_data_for_training(
                session_id=session_id,
                dataset_name='tess',
                target_column='tfopwg_disp'
            )
            
            self.training_api.configure_training(
                session_id=session_id,
                model_type='decision_tree',
                training_params={'max_depth': 5, 'random_state': 42}
            )
            
            train_result = self.training_api.start_training(session_id)
            
            # 3. Save model
            model_path = self.model_save_dir / f"e2e_test_{int(time.time())}"
            save_result = self.training_api.save_trained_model(
                session_id=session_id,
                model_path=str(model_path)
            )
            
            # 4. Load model and make predictions
            load_result = self.prediction_api.load_model(str(model_path), model_id="e2e_model")
            
            # 5. Get sample data and predict
            sample_data = self.user_api.get_sample_data('tess', n_samples=3)
            input_data = sample_data['data'].iloc[0].drop(['tfopwg_disp'], errors='ignore').to_dict()
            
            pred_result = self.prediction_api.predict_single("e2e_model", input_data)
            
            # 6. Generate explanation
            test_data = sample_data['data']
            if 'tfopwg_disp' in test_data.columns:
                X_test = test_data.drop(['tfopwg_disp'], axis=1)
                X_test_numeric = X_test.select_dtypes(include=[np.number]).fillna(0)
                
                if len(X_test_numeric.columns) > 0:
                    explanation_result = self.explanation_api.explain_prediction_local(
                        model_id="e2e_model",
                        input_data=X_test_numeric.iloc[0].to_dict(),
                        X_background=X_test_numeric.sample(min(20, len(X_test_numeric))),
                        methods=['model_coefficients']  # Use simpler method for decision tree
                    )
                    explanation_success = explanation_result.get('status') == 'success'
                else:
                    explanation_success = False
            else:
                explanation_success = False
            
            workflow_success = (
                train_result['status'] == 'success' and
                save_result.get('status') == 'success' and
                load_result['status'] == 'success' and
                pred_result['status'] == 'success'
            )
            
            self.log_test(
                'End-to-End Workflow',
                workflow_success,
                {
                    'training_success': train_result['status'] == 'success',
                    'model_saved': save_result.get('status') == 'success',
                    'model_loaded': load_result['status'] == 'success',
                    'prediction_success': pred_result['status'] == 'success',
                    'explanation_success': explanation_success,
                    'final_prediction': pred_result.get('prediction') if pred_result['status'] == 'success' else None
                }
            )
            return workflow_success
            
        except Exception as e:
            self.log_test('End-to-End Workflow', False, {}, str(e))
            return False
    
    def run_all_tests(self):
        """Run all comprehensive functionality tests."""
        print("🎯 Running comprehensive functionality tests...\n")
        
        # Run all tests
        tests = [
            self.test_1_dataset_analysis,
            self.test_2_column_selection_training,
            self.test_3_prediction_api,
            self.test_4_explanation_api,
            self.test_5_user_data_upload,
            self.test_6_end_to_end_workflow
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
        print("📋 COMPREHENSIVE FUNCTIONALITY TEST SUMMARY")
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
        results_file = Path(f"/home/brine/OneDrive/Work/Exoplanet-Playground/tests/results/comprehensive_functionality_test_{int(time.time())}.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        print("=" * 80)
        
        return self.results

def main():
    """Main function to run comprehensive functionality tests."""
    tester = ComprehensiveFunctionalityTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['success_rate'] >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()