"""
ML Integration wrapper for interfacing with existing ML APIs.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import pandas as pd
import numpy as np
import pickle

# Add ML module to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI
from ML.src.data.data_loader import DataLoader
from ML.src.data.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class MLIntegration:
    """Wrapper for ML API integration."""
    
    def __init__(self):
        """Initialize ML APIs."""
        self.training_api = TrainingAPI()
        self.prediction_api = PredictionAPI()
        self.explanation_api = ExplanationAPI(self.prediction_api)
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        
        # Target column mapping
        self.target_columns = {
            'kepler': 'koi_disposition',
            'tess': 'tfopwg_disp',
            'k2': 'disposition'
        }
        
        # Pretrained model paths
        self.pretrained_model_dir = project_root / 'ML' / 'models' / 'pretrained'
    
    # ========================================
    # CUSTOM TRAINING
    # ========================================
    
    def start_custom_training(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start custom model training.
        - For NASA datasets: Use trainval for training, test for evaluation
        - For user uploads: Use provided train/test files
        """
        try:
            # 1. Start training session
            self.training_api.start_training_session(session_id)
            
            # 2. Load training data
            if config['data_source'] == 'nasa':
                # Use trainval dataset for training
                dataset_name = config['dataset_name']
                train_data = self.data_loader.load_nasa_trainval(dataset_name)
                
                # Validate the data
                from ML.src.data.data_validator import DataValidator
                validator = DataValidator()
                validation_results = validator.validate_data_format(train_data)
                
                # Set up session properly
                session = self.training_api.current_session[session_id]
                session['data'] = train_data
                session['data_source'] = 'nasa'
                session['dataset_name'] = dataset_name
                session['data_info'] = {
                    'shape': train_data.shape,
                    'columns': list(train_data.columns),
                    'data_source': 'nasa',
                    'validation': validation_results,
                    'missing_values': train_data.isnull().sum().to_dict(),
                    'dtypes': train_data.dtypes.astype(str).to_dict()
                }
                session['status'] = 'data_loaded'
                
                # Store that we should use NASA test data later
                session['use_nasa_test'] = True
            else:
                # User uploaded files
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
            import traceback
            traceback.print_exc()
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
            
            # Determine which test data to use
            if session_info.get('use_nasa_test', False):
                # Load NASA test dataset
                dataset_name = session_info.get('dataset_name')
                test_data = self.data_loader.load_nasa_test(dataset_name)
                target_column = session_info['training_config']['target_column']
                feature_columns = session_info['training_config']['feature_columns']
                
                # Prepare test data
                X_test = test_data[feature_columns]
                if target_column in test_data.columns:
                    y_test = test_data[target_column]
                    # Apply same preprocessing as training
                    y_test = self.data_processor.create_target_variable(test_data, target_column)
                else:
                    y_test = None
                    
                logger.info(f"Using NASA test data: {len(X_test)} samples")
                
            elif 'user_test_data_path' in session_info:
                # Load user test data
                test_data = self.data_loader.load_user_dataset(session_info['user_test_data_path'])
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
                    
                logger.info(f"Using user test data: {len(X_test)} samples")
            else:
                # Use auto-split test data from training
                X_test = prepared_data['X_test']
                y_test = prepared_data['y_test']
                logger.info(f"Using auto-split test data: {len(X_test)} samples")
            
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Map prediction values to frontend-expected labels
            prediction_label_map = {
                'planet': 'Exoplanet',
                'candidate': 'Candidate',
                'false_positive': 'False Positive',
                'unknown': 'Unknown'
            }
            
            # Format ALL prediction results (removed 100 limit)
            prediction_results = []
            for i in range(len(predictions)):
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
                    'test_size': len(X_test)
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
        """
        Run pretrained model prediction.
        - For NASA datasets: Load pretrained model, predict on NASA test data
        - For user uploads: Augment NASA data with user data, train new model, predict on NASA test data
        """
        try:
            # Get dataset name
            dataset_name = config.get('dataset_name', 'kepler')
            
            # Create a session to track this
            self.training_api.start_training_session(session_id)
            session = self.training_api.current_session[session_id]
            session['dataset_name'] = dataset_name
            session['is_pretrained'] = True
            
            # Initialize progress tracking
            self.training_api.training_progress[session_id] = {
                'status': 'starting',
                'progress': 0,
                'current_step': 'Initializing pretrained prediction...'
            }
            
            if config['data_source'] == 'nasa':
                # SCENARIO 1-3: Use pretrained model for NASA datasets
                logger.info(f"Loading pretrained model for {dataset_name}")
                self.training_api._update_training_progress(session_id, 30, f'Loading pretrained model...')
                
                # Load the pretrained model
                model = self._load_pretrained_model(dataset_name)
                session['model'] = model
                
                # Load NASA test data
                self.training_api._update_training_progress(session_id, 60, f'Loading test data...')
                test_data = self.data_loader.load_nasa_test(dataset_name)
                
                # Get target column and features
                target_column = self.target_columns.get(dataset_name)
                
                # Load metadata to get feature names
                metadata = self._load_pretrained_metadata(dataset_name)
                feature_columns = metadata.get('feature_names', [])
                
                session['feature_columns'] = feature_columns
                session['target_column'] = target_column
                session['test_data'] = test_data
                
            else:
                # SCENARIO 4: User uploaded data - augment and train new model
                logger.info(f"Augmenting {dataset_name} data with user upload")
                self.training_api._update_training_progress(session_id, 20, 'Loading NASA training data...')
                
                # Load NASA trainval data
                nasa_trainval = self.data_loader.load_nasa_trainval(dataset_name)
                
                # Load user data
                self.training_api._update_training_progress(session_id, 30, 'Loading user data...')
                user_data = self.data_loader.load_user_dataset(config['train_data_path'])
                
                # Augment: combine NASA and user data
                self.training_api._update_training_progress(session_id, 40, 'Augmenting dataset...')
                augmented_data = pd.concat([nasa_trainval, user_data], ignore_index=True)
                logger.info(f"Augmented data: NASA {len(nasa_trainval)} + User {len(user_data)} = {len(augmented_data)}")
                
                # Store in session
                session['data'] = augmented_data
                session['data_source'] = 'augmented'
                session['status'] = 'data_loaded'
                
                # Configure and train
                self.training_api._update_training_progress(session_id, 50, 'Configuring training...')
                training_config = {
                    'model_type': 'random_forest',  # Use random forest for pretrained
                    'target_column': self.target_columns.get(dataset_name),
                    'hyperparameters': {'n_estimators': 100, 'max_depth': 20}
                }
                config_result = self.training_api.configure_training(session_id, training_config)
                if config_result['status'] != 'success':
                    raise ValueError(f"Configuration failed: {config_result.get('error')}")
                
                # Train the model
                self.training_api._update_training_progress(session_id, 60, 'Training model...')
                train_result = self.training_api.start_training(session_id)
                
                if train_result['status'] != 'success':
                    raise ValueError(f"Training failed: {train_result.get('error')}")
                
                # Model is now in session, load NASA test data for evaluation
                test_data = self.data_loader.load_nasa_test(dataset_name)
                session['test_data'] = test_data
                session['feature_columns'] = session['training_config']['feature_columns']
                session['target_column'] = session['training_config']['target_column']
            
            # Mark as completed
            self.training_api._update_training_progress(session_id, 100, 'Completed')
            session['status'] = 'completed'
            
            return {
                'session_id': session_id,
                'status': 'success',
                'message': 'Pretrained prediction ready'
            }
            
        except Exception as e:
            logger.error(f"Error in pretrained prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            self.training_api._update_training_progress(session_id, -1, f'Error: {str(e)}')
            return {'session_id': session_id, 'status': 'error', 'error': str(e)}
    
    def _load_pretrained_model(self, dataset_name: str):
        """Load a pretrained model from disk."""
        model_path = self.pretrained_model_dir / f'random_forest_{dataset_name}'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Pretrained model not found: {model_path}")
        
        # Load using the model's load method
        from ML.src.utils.model_factory import ModelFactory
        factory = ModelFactory()
        model = factory.create_model('random_forest')
        model.load_model(str(model_path))
        
        logger.info(f"Loaded pretrained model from {model_path}")
        return model
    
    def _load_pretrained_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Load metadata for a pretrained model."""
        metadata_path = self.pretrained_model_dir / f'random_forest_{dataset_name}_metadata.json'
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found: {metadata_path}")
            return {}
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_pretrained_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get pretrained model results (always uses NASA test data)."""
        try:
            session_result = self.training_api.get_session_info(session_id, include_data=True)
            
            if session_result['status'] != 'success':
                logger.error(f"Session result status not success: {session_result}")
                return None
            
            session_info = session_result['session_info']
            
            if session_info['status'] != 'completed':
                logger.error(f"Session status not completed: {session_info['status']}")
                return None
            
            # Get model and test data
            model = session_info['model']
            test_data = session_info['test_data']
            target_column = session_info['target_column']
            feature_columns = session_info['feature_columns']
            
            # Prepare test data
            X_test = test_data[feature_columns]
            if target_column in test_data.columns:
                y_test = self.data_processor.create_target_variable(test_data, target_column)
            else:
                y_test = None
            
            logger.info(f"Making predictions on {len(X_test)} test samples")
            
            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Calculate metrics if we have ground truth
            if y_test is not None:
                from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
                
                accuracy = accuracy_score(y_test, predictions)
                conf_matrix = confusion_matrix(y_test, predictions).tolist()
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                
                metrics = {
                    'accuracy': float(accuracy),
                    'confusion_matrix': conf_matrix,
                    'f1_score': float(f1),
                    'precision': float(precision),
                    'recall': float(recall)
                }
            else:
                metrics = {
                    'accuracy': 0,
                    'confusion_matrix': [],
                    'f1_score': 0,
                    'precision': 0,
                    'recall': 0
                }
            
            # Map prediction values to frontend-expected labels
            prediction_label_map = {
                'planet': 'Exoplanet',
                'candidate': 'Candidate',
                'false_positive': 'False Positive',
                'unknown': 'Unknown'
            }
            
            # Format ALL prediction results
            prediction_results = []
            for i in range(len(predictions)):
                instance_dict = X_test.iloc[i].to_dict()
                
                # Get top features for explanation
                feature_importance = {}
                if hasattr(model.model, 'feature_importances_'):
                    importances = model.model.feature_importances_
                    top_indices = np.argsort(importances)[-5:][::-1]
                    for idx in top_indices:
                        if idx < len(feature_columns):
                            feature_importance[feature_columns[idx]] = float(importances[idx])
                
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
                    'model_type': 'random_forest',
                    'feature_count': len(feature_columns),
                    'train_size': 0,  # Not applicable for pretrained
                    'test_size': len(X_test)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting pretrained result: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
