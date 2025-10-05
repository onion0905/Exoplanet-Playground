"""
ML Integration wrapper for interfacing with existing ML APIs.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import pandas as pd
import numpy as np

# Add ML module to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI

logger = logging.getLogger(__name__)


class MLIntegration:
    """Wrapper for ML API integration."""
    
    def __init__(self):
        """Initialize ML APIs."""
        self.training_api = TrainingAPI()
        self.prediction_api = PredictionAPI()
        self.explanation_api = ExplanationAPI(self.prediction_api)
        
        # Target column mapping
        self.target_columns = {
            'kepler': 'koi_disposition',
            'tess': 'tfopwg_disp',
            'k2': 'disposition'
        }
    
    # ========================================
    # CUSTOM TRAINING
    # ========================================
    
    def start_custom_training(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start custom model training and return training result."""
        try:
            # 1. Start training session
            self.training_api.start_training_session(session_id)
            
            # 2. Load data
            if config['data_source'] == 'nasa':
                data_config = {'datasets': [config['dataset_name']]}
                self.training_api.load_data_for_training(session_id, 'nasa', data_config)
            else:
                data_config = {'filepath': config['train_data_path']}
                self.training_api.load_data_for_training(session_id, 'user', data_config)
                
                # Store test data path for later use if provided
                if 'test_data_path' in config:
                    session = self.training_api.current_session[session_id]
                    session['user_test_data_path'] = config['test_data_path']
            
            # 3. Configure training
            dataset_name = config.get('dataset_name', 'kepler')
            training_config = {
                'model_type': config['model_type'],
                'target_column': self.target_columns.get(dataset_name, 'koi_disposition'),
                'hyperparameters': config.get('hyperparameters', {})
            }
            config_result = self.training_api.configure_training(session_id, training_config)
            if config_result['status'] != 'success':
                raise ValueError(f"Configuration failed: {config_result.get('error', 'Unknown error')}")
            
            # 4. Start training
            result = self.training_api.start_training(session_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in start_custom_training: {str(e)}")
            return {'session_id': session_id, 'status': 'error', 'error': str(e)}
    
    def get_training_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training progress from TrainingAPI."""
        try:
            if session_id not in self.training_api.training_progress:
                return None
            
            progress_info = self.training_api.training_progress[session_id]
            return {
                'progress': progress_info.get('progress', 0),
                'current_step': progress_info.get('current_step', ''),
                'status': progress_info.get('status', 'unknown')
            }
        except Exception as e:
            logger.error(f"Error getting progress: {str(e)}")
            return None
    
    def get_custom_training_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training results with validation metrics and test predictions."""
        try:
            # Get session info with data
            session_result = self.training_api.get_session_info(session_id, include_data=True)
            
            if session_result['status'] != 'success':
                logger.error(f"Session result status not success: {session_result}")
                return None
            
            session_info = session_result['session_info']
            
            if session_info['status'] != 'completed':
                logger.error(f"Session status not completed: {session_info['status']}")
                return None
            
            # Extract model and data
            model = session_info['model']
            prepared_data = session_info['prepared_data']
            
            # Validation metrics (from training)
            metrics = {
                'accuracy': float(session_info.get('validation_accuracy', 0) or 0),
                'confusion_matrix': session_info.get('evaluation_metrics', {}).get('confusion_matrix', []),
                'f1_score': float(session_info.get('evaluation_metrics', {}).get('f1_score', 0) or 0),
                'precision': float(session_info.get('evaluation_metrics', {}).get('precision', 0) or 0),
                'recall': float(session_info.get('evaluation_metrics', {}).get('recall', 0) or 0)
            }
            
            # Check if user uploaded separate test data
            if 'user_test_data_path' in session_info:
                # Load user test data
                from ML.src.data.data_loader import DataLoader
                from ML.src.data.data_processor import DataProcessor
                
                data_loader = DataLoader()
                data_processor = DataProcessor()
                
                test_data = data_loader.load_user_dataset(session_info['user_test_data_path'])
                target_column = session_info['training_config']['target_column']
                feature_columns = session_info['training_config']['feature_columns']
                
                # Prepare test data (no splitting needed)
                if target_column in test_data.columns:
                    y_test = test_data[target_column]
                    X_test = test_data[feature_columns]
                else:
                    # No target column in test data, predict only
                    X_test = test_data[feature_columns]
                    y_test = None
            else:
                # Use auto-split test data
                X_test = prepared_data['X_test']
                y_test = prepared_data['y_test']
            
            with open('d:/debug_ml.txt', 'a') as f:
                f.write(f"Test data prepared, making predictions...\n")
            
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            with open('d:/debug_ml.txt', 'a') as f:
                f.write(f"Predictions made, formatting results...\n")
            
            # Map prediction values to frontend-expected labels
            prediction_label_map = {
                'planet': 'Exoplanet',
                'candidate': 'Candidate',
                'false_positive': 'False Positive',
                'unknown': 'Unknown'
            }
            
            # Format prediction results
            prediction_results = []
            for i in range(min(len(predictions), 100)):  # Limit to 100 for performance
                instance_dict = X_test.iloc[i].to_dict()
                
                # Get top features for explanation
                feature_importance = {}
                if hasattr(model.model, 'feature_importances_'):
                    feature_names = prepared_data['feature_names']
                    importances = model.model.feature_importances_
                    # Get top 5 features
                    top_indices = np.argsort(importances)[-5:][::-1]
                    for idx in top_indices:
                        if idx < len(feature_names):
                            feature_importance[feature_names[idx]] = float(importances[idx])
                
                # Map prediction to frontend label
                raw_prediction = str(predictions[i])
                display_prediction = prediction_label_map.get(raw_prediction, raw_prediction)
                
                pred_item = {
                    'id': i,
                    'prediction': display_prediction,
                    'confidence': float(probabilities[i].max()),
                    'probabilities': {str(k): float(v) for k, v in zip(model.target_classes, probabilities[i])},
                    'feature_importance': feature_importance,
                    'features': instance_dict
                }
                
                # Add actual value if available
                if y_test is not None:
                    actual_value = str(y_test.iloc[i])
                    pred_item['actual'] = prediction_label_map.get(actual_value, actual_value)
                
                prediction_results.append(pred_item)
            
            return {
                'metrics': metrics,
                'predictions': prediction_results,
                'model_info': {
                    'model_type': session_info['training_config']['model_type'],
                    'feature_count': len(prepared_data['feature_names']),
                    'train_size': len(prepared_data['X_train']),
                    'test_size': len(prepared_data['X_test'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting training result: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # ========================================
    # PRETRAINED MODEL
    # ========================================
    
    def run_pretrained_prediction(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run pretrained model (reuse training pipeline with pretrained settings)."""
        try:
            # For simplicity, use the same training pipeline with default random_forest
            # This simulates a "pretrained" model by training quickly on the dataset
            config['model_type'] = 'random_forest'
            config['hyperparameters'] = {'n_estimators': 50, 'max_depth': 10}  # Fast training
            
            return self.start_custom_training(session_id, config)
            
        except Exception as e:
            logger.error(f"Error in pretrained prediction: {str(e)}")
            return {'session_id': session_id, 'status': 'error', 'error': str(e)}
    
    def get_pretrained_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get pretrained model results (same as custom training)."""
        return self.get_custom_training_result(session_id)

