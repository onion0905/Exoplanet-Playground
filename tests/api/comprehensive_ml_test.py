#!/usr/bin/env python3
"""
Comprehensive ML API Test Suite - All Model/Dataset Combinations

This script tests all possible combinations of:
- 7 Model Types: random_forest, decision_tree, linear_regression, svm, xgboost, pca, deep_learning
- 3 Datasets: kepler, tess, k2
- 3 Operations: training, prediction, explanation

Total combinations to test: 7 × 3 × 3 = 63 test cases

Usage:
    source .venv/bin/activate.fish
    python comprehensive_ml_test.py
"""

import sys
import os
import time
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Tuple

# Add ML directory to Python path
ml_dir = Path(__file__).parent.parent.parent / "ML"
sys.path.insert(0, str(ml_dir))

# Import ML APIs
from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI
from ML.src.utils.model_factory import ModelFactory

class ComprehensiveMLTest:
    """Comprehensive test suite for all ML API combinations"""
    
    def __init__(self):
        self.api = ExoplanetMLAPI()
        self.training_api = TrainingAPI()
        self.prediction_api = PredictionAPI()
        self.explanation_api = ExplanationAPI()
        self.model_factory = ModelFactory()
        
        # Test configurations
        self.datasets = ['kepler', 'tess', 'k2']
        self.model_types = ['random_forest', 'decision_tree', 'linear_regression', 
                           'svm', 'xgboost', 'pca', 'deep_learning']
        
        # Results tracking
        self.results = {
            'training': {},
            'prediction': {},
            'explanation': {},
            'overall_stats': {}
        }
        
        # Test parameters
        self.max_training_time = 300  # 5 minutes per model
        self.sample_size = 1000  # Use smaller sample for faster testing
        
    def print_header(self, title: str, level: int = 1):
        """Print formatted header"""
        symbols = ["=" * 80, "-" * 60, "~" * 40]
        symbol = symbols[min(level - 1, 2)]
        print(f"\n{symbol}")
        print(f"{'🧪' if level == 1 else '📋' if level == 2 else '🔍'} {title.upper()}")
        print(symbol)
    
    def test_data_loading(self) -> bool:
        """Test that all datasets can be loaded"""
        self.print_header("Testing Data Loading", 2)
        
        for dataset in self.datasets:
            try:
                info = self.api.get_dataset_info(dataset)
                if 'error' in info:
                    print(f"❌ Dataset {dataset}: {info['error']}")
                    return False
                else:
                    print(f"✅ Dataset {dataset}: {info['total_records']} records, {info['original_features']} features")
            except Exception as e:
                print(f"❌ Dataset {dataset}: {str(e)}")
                return False
                
        return True
    
    def train_model_combination(self, model_type: str, dataset: str) -> Tuple[bool, str, Dict]:
        """Train a specific model-dataset combination"""
        print(f"🔄 Training {model_type} on {dataset}...")
        
        try:
            # Start training session
            session_id = f"test_{model_type}_{dataset}_{int(time.time())}"
            start_time = time.time()
            
            # 1. Start training session
            session_info = self.training_api.start_training_session(session_id)
            if session_info.get('status') != 'initialized':
                return False, f"Failed to initialize training session: {session_info}", {}
            
            # 2. Load data
            data_config = {'datasets': [dataset]}
            load_result = self.training_api.load_data_for_training(
                session_id, 
                'nasa',  # data_source
                data_config
            )
            
            if 'error' in load_result:
                return False, f"Data loading failed: {load_result['error']}", {}
            
            # 3. Configure training
            # Determine target column based on dataset
            target_column_map = {
                'kepler': 'koi_disposition',
                'tess': 'tfopwg_disp',  # Use tfopwg_disp as it has more data
                'k2': 'disposition'
            }
            target_column = target_column_map.get(dataset)
            
            training_config = {
                'model_type': model_type,
                'target_column': target_column,
                'hyperparameters': self._get_fast_hyperparameters(model_type)
            }
            
            config_result = self.training_api.configure_training(session_id, training_config)
            if 'error' in config_result:
                return False, f"Configuration failed: {config_result['error']}", {}
            
            # 4. Start training
            training_result = self.training_api.start_training(session_id)
            
            training_time = time.time() - start_time
            
            if training_result.get('status') == 'completed':
                model_path = training_result.get('model_path', '')
                metrics = training_result.get('metrics', {})
                
                return True, f"Training completed in {training_time:.1f}s", {
                    'model_path': model_path,
                    'training_time': training_time,
                    'metrics': metrics,
                    'session_id': session_id
                }
            else:
                return False, f"Training failed: {training_result.get('message', 'Unknown error')}", {}
                
        except Exception as e:
            error_msg = f"Training exception: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg, {}
    
    def _get_fast_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters optimized for fast testing"""
        if model_type == 'deep_learning':
            return {
                'epochs': 5,
                'batch_size': 32,
                'hidden_layers': [32, 16]
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': 10,
                'max_depth': 5
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': 10,
                'max_depth': 3
            }
        else:
            return {}
    
    def test_prediction(self, model_info: Dict, dataset: str) -> Tuple[bool, str]:
        """Test prediction using a trained model"""
        try:
            model_path = model_info.get('model_path')
            if not model_path:
                return False, "No model path available"
            
            # Load the model for prediction
            load_result = self.prediction_api.load_model(model_path)
            if 'error' in load_result:
                return False, f"Model loading failed: {load_result['error']}"
            
            # Get sample data for prediction
            sample_data = self.api.get_sample_data(dataset, n_samples=5)
            if 'error' in sample_data:
                return False, f"Sample data failed: {sample_data['error']}"
            
            # Make predictions
            predictions = self.prediction_api.predict(
                model_path, 
                sample_data['sample_data']
            )
            
            if 'error' in predictions:
                return False, f"Prediction failed: {predictions['error']}"
            
            return True, f"Predicted {len(predictions.get('predictions', []))} samples"
            
        except Exception as e:
            return False, f"Prediction exception: {str(e)}"
    
    def test_explanation(self, model_info: Dict, dataset: str) -> Tuple[bool, str]:
        """Test explanation generation for a trained model"""
        try:
            model_path = model_info.get('model_path')
            if not model_path:
                return False, "No model path available"
            
            # Test feature importance
            importance_result = self.explanation_api.get_feature_importance(model_path)
            if 'error' in importance_result:
                return False, f"Feature importance failed: {importance_result['error']}"
            
            # Test column dropping analysis (if supported)
            try:
                drop_result = self.explanation_api.analyze_column_drop_impact(
                    model_path, 
                    dataset
                )
                if 'error' not in drop_result:
                    return True, f"Generated explanations: importance + drop analysis"
            except:
                pass
            
            return True, "Generated feature importance"
            
        except Exception as e:
            return False, f"Explanation exception: {str(e)}"
    
    def run_training_tests(self):
        """Run all training combinations"""
        self.print_header("Training Tests - All Model/Dataset Combinations", 2)
        
        total_combinations = len(self.model_types) * len(self.datasets)
        current = 0
        
        for model_type in self.model_types:
            for dataset in self.datasets:
                current += 1
                print(f"\n[{current}/{total_combinations}] Testing {model_type} + {dataset}")
                
                success, message, model_info = self.train_model_combination(model_type, dataset)
                
                # Store results
                key = f"{model_type}_{dataset}"
                self.results['training'][key] = {
                    'success': success,
                    'message': message,
                    'model_info': model_info,
                    'model_type': model_type,
                    'dataset': dataset
                }
                
                if success:
                    print(f"✅ {message}")
                else:
                    print(f"❌ {message}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(1)
    
    def run_prediction_tests(self):
        """Run prediction tests on successfully trained models"""
        self.print_header("Prediction Tests", 2)
        
        trained_models = [(k, v) for k, v in self.results['training'].items() if v['success']]
        
        if not trained_models:
            print("❌ No successfully trained models to test predictions")
            return
        
        for key, training_result in trained_models:
            model_info = training_result['model_info']
            dataset = training_result['dataset']
            model_type = training_result['model_type']
            
            print(f"\n🔍 Testing prediction: {model_type} on {dataset}")
            
            success, message = self.test_prediction(model_info, dataset)
            
            self.results['prediction'][key] = {
                'success': success,
                'message': message,
                'model_type': model_type,
                'dataset': dataset
            }
            
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
    
    def run_explanation_tests(self):
        """Run explanation tests on successfully trained models"""
        self.print_header("Explanation Tests", 2)
        
        trained_models = [(k, v) for k, v in self.results['training'].items() if v['success']]
        
        if not trained_models:
            print("❌ No successfully trained models to test explanations")
            return
        
        for key, training_result in trained_models:
            model_info = training_result['model_info']
            dataset = training_result['dataset']
            model_type = training_result['model_type']
            
            print(f"\n🔬 Testing explanation: {model_type} on {dataset}")
            
            success, message = self.test_explanation(model_info, dataset)
            
            self.results['explanation'][key] = {
                'success': success,
                'message': message,
                'model_type': model_type,
                'dataset': dataset
            }
            
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        self.print_header("Test Results Summary", 1)
        
        # Calculate statistics
        total_combinations = len(self.model_types) * len(self.datasets)
        
        training_success = len([r for r in self.results['training'].values() if r['success']])
        prediction_success = len([r for r in self.results['prediction'].values() if r['success']])
        explanation_success = len([r for r in self.results['explanation'].values() if r['success']])
        
        print(f"📊 OVERALL STATISTICS")
        print(f"   Total model/dataset combinations: {total_combinations}")
        print(f"   ✅ Training successful: {training_success}/{total_combinations} ({training_success/total_combinations*100:.1f}%)")
        print(f"   ✅ Prediction successful: {prediction_success}/{training_success if training_success > 0 else 1} ({prediction_success/training_success*100 if training_success > 0 else 0:.1f}%)")
        print(f"   ✅ Explanation successful: {explanation_success}/{training_success if training_success > 0 else 1} ({explanation_success/training_success*100 if training_success > 0 else 0:.1f}%)")
        
        # Results by model type
        print(f"\n📈 RESULTS BY MODEL TYPE")
        for model_type in self.model_types:
            model_results = [r for k, r in self.results['training'].items() if r['model_type'] == model_type]
            successes = len([r for r in model_results if r['success']])
            total = len(model_results)
            print(f"   {model_type:15} : {successes}/{total} ({successes/total*100:.1f}%)")
        
        # Results by dataset
        print(f"\n📊 RESULTS BY DATASET")
        for dataset in self.datasets:
            dataset_results = [r for k, r in self.results['training'].items() if r['dataset'] == dataset]
            successes = len([r for r in dataset_results if r['success']])
            total = len(dataset_results)
            print(f"   {dataset:8} : {successes}/{total} ({successes/total*100:.1f}%)")
        
        # Failed combinations
        failed_training = [(k, r) for k, r in self.results['training'].items() if not r['success']]
        if failed_training:
            print(f"\n❌ FAILED TRAINING COMBINATIONS")
            for key, result in failed_training:
                print(f"   {result['model_type']} + {result['dataset']}: {result['message']}")
        
        # Save detailed results
        self.save_detailed_report()
        
        print(f"\n🎯 FINAL ASSESSMENT")
        if training_success == total_combinations:
            print("🎉 EXCELLENT! All model/dataset combinations trained successfully")
        elif training_success > total_combinations * 0.8:
            print("✅ GOOD! Most combinations working well")
        elif training_success > total_combinations * 0.5:
            print("⚠️  PARTIAL! Some combinations need attention")
        else:
            print("❌ NEEDS WORK! Many combinations failing")
    
    def save_detailed_report(self):
        """Save detailed results to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"ml_test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE ML API TEST REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TRAINING RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, result in self.results['training'].items():
                f.write(f"{key}: {'SUCCESS' if result['success'] else 'FAILED'}\n")
                f.write(f"  Message: {result['message']}\n")
                if result['success'] and 'training_time' in result['model_info']:
                    f.write(f"  Training Time: {result['model_info']['training_time']:.1f}s\n")
                f.write("\n")
            
            f.write("\nPREDICTION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, result in self.results['prediction'].items():
                f.write(f"{key}: {'SUCCESS' if result['success'] else 'FAILED'}\n")
                f.write(f"  Message: {result['message']}\n\n")
            
            f.write("\nEXPLANATION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, result in self.results['explanation'].items():
                f.write(f"{key}: {'SUCCESS' if result['success'] else 'FAILED'}\n")
                f.write(f"  Message: {result['message']}\n\n")
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        start_time = time.time()
        
        self.print_header("Comprehensive ML API Test Suite", 1)
        print(f"Testing {len(self.model_types)} model types × {len(self.datasets)} datasets")
        print(f"Total combinations: {len(self.model_types) * len(self.datasets)}")
        print(f"Operations per combination: Training + Prediction + Explanation")
        
        # Test data loading first
        if not self.test_data_loading():
            print("❌ Data loading tests failed. Aborting.")
            return
        
        try:
            # Run all test phases
            self.run_training_tests()
            self.run_prediction_tests()
            self.run_explanation_tests()
            
            # Generate final report
            self.generate_report()
            
            total_time = time.time() - start_time
            print(f"\n⏱️  Total test time: {total_time/60:.1f} minutes")
            
        except KeyboardInterrupt:
            print("\n🛑 Tests interrupted by user")
            self.generate_report()
        except Exception as e:
            print(f"\n💥 Test suite error: {str(e)}")
            traceback.print_exc()

def main():
    """Main function"""
    print("🚀 Starting Comprehensive ML API Test Suite")
    print("This will test all model types with all datasets")
    print("⚠️  This may take 15-30 minutes depending on your system")
    
    response = input("\nProceed with comprehensive testing? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Test cancelled.")
        return
    
    test_suite = ComprehensiveMLTest()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()