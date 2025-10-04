from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

from ..data.data_loader import DataLoader
from ..data.data_processor import DataProcessor
from ..data.data_validator import DataValidator
from ..utils.model_factory import ModelFactory


class TrainingAPI:
    """API interface for training exoplanet ML models."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Use centralized config
            from ..config import DATA_DIR
            data_dir = DATA_DIR
            
        self.data_loader = DataLoader(data_dir)
        self.data_processor = DataProcessor()
        self.data_validator = DataValidator()
        self.model_factory = ModelFactory()
        self.logger = logging.getLogger(__name__)
        
        # Training session state
        self.current_session = {}
        self.training_progress = {}
        
    def start_training_session(self, session_id: str) -> Dict[str, Any]:
        """Start a new training session."""
        self.current_session[session_id] = {
            'created_at': datetime.now().isoformat(),
            'status': 'initialized',
            'model': None,
            'data_info': None,
            'training_config': None
        }
        
        return {
            'session_id': session_id,
            'status': 'initialized',
            'message': 'Training session started'
        }
    
    def load_data_for_training(self, session_id: str, 
                             data_source: str,
                             data_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load data for training.
        
        Args:
            session_id: Training session ID
            data_source: 'nasa' or 'user'  
            data_config: Configuration for data loading
        """
        try:
            if data_source == 'nasa':
                # Load NASA dataset(s)
                dataset_names = data_config.get('datasets', ['kepler'])
                if len(dataset_names) == 1:
                    data = self.data_loader.load_nasa_dataset(dataset_names[0])
                else:
                    data = self.data_loader.combine_datasets(dataset_names)
                    
            elif data_source == 'user':
                # Load user-uploaded data
                filepath = data_config.get('filepath')
                if not filepath:
                    raise ValueError("User data requires 'filepath' in config")
                data = self.data_loader.load_user_dataset(filepath)
            else:
                raise ValueError(f"Unknown data source: {data_source}")
            
            # Validate data
            validation_results = self.data_validator.validate_data_format(data)
            
            # Store data info in session
            data_info = {
                'shape': data.shape,
                'columns': list(data.columns),
                'data_source': data_source,
                'validation': validation_results,
                'missing_values': data.isnull().sum().to_dict(),
                'dtypes': data.dtypes.astype(str).to_dict()
            }
            
            self.current_session[session_id]['data'] = data
            self.current_session[session_id]['data_info'] = data_info
            self.current_session[session_id]['status'] = 'data_loaded'
            
            return {
                'session_id': session_id,
                'status': 'success',
                'data_info': data_info
            }
            
        except Exception as e:
            self.logger.error(f"Error loading data for session {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def configure_training(self, session_id: str, 
                         training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure training parameters.
        
        Args:
            session_id: Training session ID
            training_config: Dictionary containing:
                - model_type: str (e.g., 'random_forest', 'xgboost')
                - target_column: str
                - feature_columns: List[str] (optional)
                - hyperparameters: Dict[str, Any] (optional)
                - preprocessing_config: Dict[str, Any] (optional)
        """
        try:
            if session_id not in self.current_session:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.current_session[session_id]
            if 'data' not in session:
                raise ValueError("Data must be loaded before configuring training")
            
            # Validate target column
            target_column = training_config.get('target_column')
            if not target_column:
                raise ValueError("target_column is required")
            
            data = session['data']
            target_validation = self.data_validator.validate_target_column(data, target_column)
            
            # Get recommended features if not specified
            feature_columns = training_config.get('feature_columns')
            if not feature_columns:
                dataset_type = session['data_info']['validation'].get('dataset_type')
                feature_columns = self.data_validator.get_recommended_features(data, dataset_type)
                self.logger.info(f"Using {len(feature_columns)} recommended features")
            
            # Store training configuration
            session['training_config'] = {
                'model_type': training_config['model_type'],
                'target_column': target_column,
                'feature_columns': feature_columns,
                'hyperparameters': training_config.get('hyperparameters', {}),
                'preprocessing_config': training_config.get('preprocessing_config', {}),
                'target_validation': target_validation
            }
            
            session['status'] = 'configured'
            
            return {
                'session_id': session_id,
                'status': 'success',
                'training_config': session['training_config']
            }
            
        except Exception as e:
            self.logger.error(f"Error configuring training for session {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def start_training(self, session_id: str) -> Dict[str, Any]:
        """Start the training process."""
        try:
            if session_id not in self.current_session:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.current_session[session_id]
            if session['status'] != 'configured':
                raise ValueError("Training must be configured before starting")
            
            # Initialize progress tracking
            self.training_progress[session_id] = {
                'status': 'starting',
                'progress': 0,
                'current_step': 'Initializing training...'
            }
            
            # Get data and config
            data = session['data']
            config = session['training_config']
            
            # Update progress
            self._update_training_progress(session_id, 10, 'Preparing data...')
            
            # Prepare data
            prepared_data = self.data_processor.prepare_data(
                data=data,
                target_column=config['target_column'],
                feature_columns=config['feature_columns'],
                preprocessing_config=config['preprocessing_config']
            )
            
            # Update progress
            self._update_training_progress(session_id, 30, 'Creating model...')
            
            # Create model
            model = self.model_factory.create_model(config['model_type'])
            model.build_model(**config['hyperparameters'])
            
            # Update progress
            self._update_training_progress(session_id, 40, 'Training model...')
            
            # Train model
            training_metrics = model.train(
                X_train=prepared_data['X_train'],
                y_train=prepared_data['y_train'],
                X_val=prepared_data.get('X_val'),
                y_val=prepared_data.get('y_val')
            )
            
            # Update progress
            self._update_training_progress(session_id, 80, 'Evaluating model...')
            
            # Evaluate on test set
            evaluation_metrics = model.evaluate(
                prepared_data['X_test'],
                prepared_data['y_test']
            )
            
            # Store results
            session['model'] = model
            session['prepared_data'] = prepared_data
            session['training_metrics'] = training_metrics
            session['evaluation_metrics'] = evaluation_metrics
            session['status'] = 'completed'
            
            # Update progress
            self._update_training_progress(session_id, 100, 'Training completed!')
            
            return {
                'session_id': session_id,
                'status': 'success',
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error training model for session {session_id}: {str(e)}")
            self._update_training_progress(session_id, -1, f'Error: {str(e)}')
            return {
                'session_id': session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_training_progress(self, session_id: str) -> Dict[str, Any]:
        """Get current training progress."""
        if session_id not in self.training_progress:
            return {
                'session_id': session_id,
                'status': 'not_found',
                'error': 'Training session not found'
            }
        
        return {
            'session_id': session_id,
            'status': 'success',
            'progress': self.training_progress[session_id]
        }
    
    def save_trained_model(self, session_id: str, 
                          model_name: str,
                          save_path: str = None) -> Dict[str, Any]:
        """Save a trained model."""
        try:
            if session_id not in self.current_session:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.current_session[session_id]
            if 'model' not in session or session['status'] != 'completed':
                raise ValueError("No trained model found in session")
            
            # Default save path
            if save_path is None:
                from ..config import MODEL_SAVE_DIR
                MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
                save_path = str(MODEL_SAVE_DIR / model_name)
            
            # Save model
            model = session['model']
            model.save_model(save_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': session['training_config']['model_type'],
                'training_config': session['training_config'],
                'training_metrics': session['training_metrics'],
                'evaluation_metrics': session['evaluation_metrics'],
                'saved_at': datetime.now().isoformat()
            }
            
            metadata_path = f"{save_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'session_id': session_id,
                'status': 'success',
                'model_path': save_path,
                'metadata_path': metadata_path
            }
            
        except Exception as e:
            self.logger.error(f"Error saving model for session {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a training session."""
        if session_id not in self.current_session:
            return {
                'session_id': session_id,
                'status': 'not_found',
                'error': 'Session not found'
            }
        
        session = self.current_session[session_id].copy()
        
        # Remove large data objects for API response
        if 'data' in session:
            del session['data']
        if 'prepared_data' in session:
            del session['prepared_data']
        if 'model' in session:
            session['model'] = session['model'].get_model_info()
        
        return {
            'session_id': session_id,
            'status': 'success',
            'session_info': session
        }
    
    def _update_training_progress(self, session_id: str, progress: int, step: str):
        """Update training progress."""
        if session_id in self.training_progress:
            self.training_progress[session_id].update({
                'progress': progress,
                'current_step': step,
                'updated_at': datetime.now().isoformat()
            })
            if progress == 100:
                self.training_progress[session_id]['status'] = 'completed'
            elif progress == -1:
                self.training_progress[session_id]['status'] = 'error'
            else:
                self.training_progress[session_id]['status'] = 'running'