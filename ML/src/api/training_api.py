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
    def get_validation_confusion_matrix(self, session_id: str):
        """Return confusion matrix for validation set, with labels ['planet', 'candidate', 'false_positive'] in that order if present."""
        session = self.current_session[session_id]
        prepared = session.get('prepared_data', {})
        model = session.get('model')
        if 'X_val' in prepared and prepared['X_val'] is not None:
            from sklearn.metrics import confusion_matrix
            y_val = prepared['y_val']
            y_pred = model.predict(prepared['X_val'])
            # Standard order for exoplanet classification
            label_order = ['planet', 'candidate', 'false_positive']
            # Only include labels present in the data
            present_labels = [lbl for lbl in label_order if lbl in set(y_val)]
            cm = confusion_matrix(y_val, y_pred, labels=present_labels)
            return {'labels': present_labels, 'confusion_matrix': cm.tolist()}
        return None
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

            # --- FILTER: Only keep CONFIRMED and FALSE POSITIVE ---


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
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data columns: {list(data.columns)}")
            target_validation = self.data_validator.validate_target_column(data, target_column)

            # Get recommended features if not specified
            feature_columns = training_config.get('feature_columns')
            if feature_columns is None:
                dataset_type = session['data_info']['validation'].get('dataset_type')
                feature_columns = self.data_validator.get_recommended_features(data, dataset_type)
                self.logger.info(f"Using {len(feature_columns)} recommended features")
            # Validate feature columns are not empty
            if feature_columns is None or not isinstance(feature_columns, list) or len(feature_columns) == 0:
                raise ValueError("No feature columns specified for training.")

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
            # Always use a validation split if data is large enough
            n_rows = len(data)
            val_size = 0.18 if n_rows > 100 else 0.0  # Use 18% for validation if enough data
            prepared_data = self.data_processor.prepare_data(
                data=data,
                target_column=config['target_column'],
                feature_columns=config['feature_columns'],
                val_size=val_size,
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
            # Compute validation accuracy if validation set exists
            val_accuracy = None
            if 'X_val' in prepared_data and prepared_data['X_val'] is not None:
                val_preds = model.predict(prepared_data['X_val'])
                val_true = prepared_data['y_val']
                if hasattr(model, 'score'):
                    val_accuracy = model.score(prepared_data['X_val'], val_true)
                else:
                    val_accuracy = (val_preds == val_true).mean()
            # Store results
            session['model'] = model
            # Always retain prepared_data in the session for downstream API use
            session['prepared_data'] = prepared_data
            session['training_metrics'] = training_metrics
            session['evaluation_metrics'] = evaluation_metrics
            session['validation_accuracy'] = val_accuracy
            session['status'] = 'completed'
            # Update progress
            self._update_training_progress(session_id, 100, 'Training completed!')
            return {
                'session_id': session_id,
                'status': 'success',
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_metrics,
                'validation_accuracy': val_accuracy
            }
        except Exception as e:
            self.logger.error(f"Error training model for session {session_id}: {str(e)}")
            self._update_training_progress(session_id, -1, f'Error: {str(e)}')
            return {
                'session_id': session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_session_info(self, session_id: str, include_data: bool = False) -> Dict[str, Any]:
        """Get information about a training session. Set include_data=True to include filtered data (for testing)."""
        if session_id not in self.current_session:
            return {
                'session_id': session_id,
                'status': 'not_found',
                'error': 'Session not found'
            }
        session = self.current_session[session_id].copy()
        # Remove large data objects for API response unless requested
        if not include_data:
            if 'data' in session:
                del session['data']
            if 'prepared_data' in session:
                del session['prepared_data']
            # Convert model to dict only when include_data=False (for API responses)
            if 'model' in session and hasattr(session['model'], 'get_model_info'):
                session['model'] = session['model'].get_model_info()
        # When include_data=True, keep model object and prepared_data for predictions
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
    
    def save_trained_model(self, session_id: str, model_name: str, save_path: str = None) -> Dict[str, Any]:
        """Save a trained model and its metadata."""
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
            import numpy as np
            def convert_np(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj
            metadata_path = f"{save_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=convert_np)
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
    
    def quick_configure_training(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """
        Simplified method to configure training with a DataFrame directly.
        
        Args:
            data: Input DataFrame
            config: Configuration dictionary
            
        Returns:
            session_id for the configured training
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        # Apply NASA API filtering if requested
        if config.get('use_nasa_filtering', False):
            data, excluded_cols = self.data_processor.apply_nasa_api_filtering(data)
            self.logger.info(f"Applied NASA API filtering, excluded {len(excluded_cols)} columns")
        
        # Start session and store data
        self.start_training_session(session_id)
        self.current_session[session_id]['data'] = data
        self.current_session[session_id]['data_info'] = {
            'shape': data.shape,
            'validation': {'dataset_type': 'auto-detected'}
        }
        
        # Configure training
        training_config = {
            'model_type': config.get('model_type', 'random_forest'),
            'target_column': config['target_column'],
            'feature_columns': config.get('feature_columns'),
            'hyperparameters': config.get('hyperparameters', {}),
            'preprocessing_config': config.get('preprocessing_config', {})
        }
        
        self.configure_training(session_id, training_config)
        
        return session_id
    
    def quick_train_model(self, session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplified method to train a model.
        
        Args:
            session_id: Session ID from configure_training
            model_config: Model configuration
            
        Returns:
            Training results
        """
        # Update session with model config
        if session_id not in self.current_session:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.current_session[session_id]
        session['training_config']['model_type'] = model_config.get('model_type', 'random_forest')
        
        # Update hyperparameters, excluding non-hyperparameter keys
        hyperparams = {k: v for k, v in model_config.items() if k != 'model_type'}
        session['training_config']['hyperparameters'].update(hyperparams)
        
        # Start training
        return self.start_training(session_id)