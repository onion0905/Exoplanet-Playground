"""
Flask backend for the Exoplanet Machine Learning Web Application.
This application allows users to train ML models to identify exoplanets.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import uuid
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Use mock ML APIs for web development and testing
# The actual ML integration can be done later after web functionality is validated
class MockTrainingAPI:
    def __init__(self):
        self.sessions = {}
    
    def load_data_for_training(self, dataset_name, file_path=None):
        return {'status': 'success', 'message': f'Loaded {dataset_name} data', 'data_info': {'rows': 1000, 'features': 20}}
    
    def configure_training(self, session_id, config):
        return {'status': 'success', 'message': 'Training configured', 'config': config}
    
    def start_training(self, session_id):
        return {'status': 'success', 'message': 'Training started', 'progress': 0}
    
    def get_training_progress(self, session_id):
        return {'status': 'success', 'progress': 75, 'metrics': {'accuracy': 0.85}}
    
    def get_session_info(self, session_id):
        return {'status': 'success', 'session_id': session_id, 'model_trained': True}
    
    def save_trained_model(self, session_id, model_name):
        return {'status': 'success', 'model_path': f'/models/{model_name}.joblib', 'model_name': model_name}

class MockPredictionAPI:
    def __init__(self):
        pass
    
    def load_model(self, model_path, session_id):
        return {'status': 'success', 'message': 'Model loaded successfully'}
    
    def make_predictions(self, session_id, data):
        return {
            'status': 'success', 
            'predictions': [{'id': i, 'prediction': 'Exoplanet', 'confidence': 0.85 + i*0.01} for i in range(5)],
            'summary': {'total': 5, 'exoplanets': 4, 'false_positives': 1}
        }

class MockExplanationAPI:
    def __init__(self):
        pass
    
    def generate_explanation(self, session_id, instance_id, explanation_type='local'):
        return {
            'status': 'success',
            'explanation': {
                'type': explanation_type,
                'features': ['feature_1', 'feature_2', 'feature_3'],
                'importance': [0.6, 0.3, 0.1],
                'explanation_text': f'Mock explanation for instance {instance_id}'
            }
        }

# Initialize APIs
training_api = MockTrainingAPI()
prediction_api = MockPredictionAPI()
explanation_api = MockExplanationAPI()

# Initialize Flask app
app = Flask(__name__)
# Configure CORS - Allow requests from frontend
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001', 'http://127.0.0.1:3001'])  # Enable CORS for frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML APIs are initialized above with mock classes

# Global storage for session management (in production, use Redis or database)
active_sessions = {}
training_sessions = {}

# Configuration
UPLOAD_FOLDER = Path(__file__).parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/', methods=['GET'])
def home():
    """
    Homepage API endpoint - provides project introduction and information.
    Returns JSON with project details for the frontend to display.
    """
    try:
        return jsonify({
            'status': 'success',
            'title': 'Exoplanet Discovery Playground',
            'description': 'Train your own machine learning models to identify exoplanets using NASA data',
            'message': 'Welcome to the Exoplanet Machine Learning Platform!',
            'features': [
                'Interactive model training with multiple ML algorithms',
                'NASA Kepler, K2, and TESS datasets',
                'Custom data upload support',
                'Hyperparameter tuning',
                'Model explanation and visualization',
                'Real-time training progress tracking'
            ],
            'datasets_available': ['Kepler', 'K2', 'TESS'],
            'models_available': [
                'Linear Regression', 'SVM', 'Decision Tree', 
                'Random Forest', 'XGBoost', 'PCA', 'Neural Network'
            ],
            'getting_started': {
                'step1': 'Select your dataset and model',
                'step2': 'Configure hyperparameters', 
                'step3': 'Train your model',
                'step4': 'Make predictions and view results'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in home endpoint: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/datasets', methods=['GET'])
def get_available_datasets():
    """Get information about available NASA datasets."""
    try:
        datasets_info = {
            'kepler': {
                'name': 'Kepler Objects of Interest',
                'description': 'Planet candidates and confirmed planets from the Kepler mission',
                'features': ['orbital_period', 'transit_duration', 'depth', 'stellar_magnitude'],
                'target': 'koi_disposition',
                'classes': ['CONFIRMED', 'FALSE POSITIVE']
            },
            'k2': {
                'name': 'K2 Planets and Candidates', 
                'description': 'Extended Kepler mission observations',
                'features': ['period', 'duration', 'depth', 'star_radius'],
                'target': 'disposition',
                'classes': ['CONFIRMED', 'FALSE POSITIVE']
            },
            'tess': {
                'name': 'TESS Objects of Interest',
                'description': 'Transiting planet candidates from TESS mission',
                'features': ['orbital_period', 'transit_depth', 'stellar_radius', 'stellar_mass'],
                'target': 'tfopwg_disp',
                'classes': ['CP', 'FP']  # Confirmed Planet, False Positive
            }
        }
        
        return jsonify({
            'status': 'success',
            'datasets': datasets_info,
            'total_datasets': len(datasets_info)
        })
        
    except Exception as e:
        logger.error(f"Error getting datasets info: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get information about available ML models."""
    try:
        models_info = {
            'linear_regression': {
                'name': 'Linear Regression',
                'description': 'Simple linear model for baseline predictions',
                'hyperparameters': ['fit_intercept', 'normalize'],
                'suitable_for': 'Regression tasks and baseline comparisons'
            },
            'svm': {
                'name': 'Support Vector Machine',
                'description': 'Powerful classification with kernel methods',
                'hyperparameters': ['C', 'kernel', 'gamma'],
                'suitable_for': 'Non-linear classification problems'
            },
            'decision_tree': {
                'name': 'Decision Tree',
                'description': 'Interpretable tree-based classifier',
                'hyperparameters': ['max_depth', 'min_samples_split', 'criterion'],
                'suitable_for': 'Interpretable models with clear decision rules'
            },
            'random_forest': {
                'name': 'Random Forest',
                'description': 'Ensemble of decision trees',
                'hyperparameters': ['n_estimators', 'max_depth', 'min_samples_split'],
                'suitable_for': 'High accuracy with good generalization'
            },
            'xgboost': {
                'name': 'XGBoost',
                'description': 'Gradient boosting framework',
                'hyperparameters': ['learning_rate', 'max_depth', 'n_estimators'],
                'suitable_for': 'Competitive machine learning with high performance'
            },
            'neural_network': {
                'name': 'Neural Network',
                'description': 'Multi-layer perceptron classifier',
                'hyperparameters': ['hidden_layer_sizes', 'activation', 'learning_rate'],
                'suitable_for': 'Complex pattern recognition'
            }
        }
        
        return jsonify({
            'status': 'success',
            'models': models_info,
            'total_models': len(models_info)
        })
        
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/select', methods=['GET', 'POST'])
def select_model_and_data():
    """
    Select endpoint - handles model and dataset selection, hyperparameter configuration.
    GET: Returns session info, available datasets/models, and current configuration
    POST: Updates session configuration based on frontend workflow
    """
    try:
        if request.method == 'GET':
            # Return session info or create new session
            session_id = request.args.get('session_id')
            
            if not session_id:
                # Create new session
                session_id = str(uuid.uuid4())
                active_sessions[session_id] = {
                    'created_at': datetime.now().isoformat(),
                    'status': 'initialized',
                    'current_step': 1,
                    'data_source': 'nasa',  # Default to NASA
                    'selected_dataset': 'kepler',  # Default dataset
                    'selected_model': '',
                    'uploaded_file_info': None,
                    'hyperparameters': {},
                    'step_progress': {
                        'step1_complete': False,  # Data selection
                        'step2_complete': False,  # Model selection  
                        'step3_complete': False   # Hyperparameters
                    }
                }
            
            session = active_sessions.get(session_id, {})
            
            # Get available datasets and models info
            datasets_info = {
                'kepler': {
                    'id': 'kepler',
                    'name': 'Kepler Dataset',
                    'description': 'Objects of Interest from the Kepler mission',
                    'img': '/kepler.png',
                    'color': '#fbbf24'
                },
                'k2': {
                    'id': 'k2',
                    'name': 'K2 Dataset',
                    'description': 'Extended Kepler mission data',
                    'img': '/k2.png',
                    'color': '#ef4444'
                },
                'tess': {
                    'id': 'tess',
                    'name': 'TESS Dataset',
                    'description': 'Transiting Exoplanet Survey Satellite data',
                    'img': '/tess.png',
                    'color': '#3b82f6'
                }
            }
            
            models_info = [
                {'value': 'linear_regression', 'label': 'Linear Regression'},
                {'value': 'svm', 'label': 'Support Vector Machine (SVM)'},
                {'value': 'decision_tree', 'label': 'Decision Tree'},
                {'value': 'random_forest', 'label': 'Random Forest'},
                {'value': 'xgboost', 'label': 'XGBoost'},
                {'value': 'pca', 'label': 'Principal Component Analysis (PCA)'},
                {'value': 'neural_network', 'label': 'Neural Network'}
            ]
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'session_info': session,
                'datasets': datasets_info,
                'models': models_info,
                'current_step': session.get('current_step', 1),
                'max_step': _calculate_max_step(session),
                'is_training_ready': _is_training_ready(session)
            })
            
        elif request.method == 'POST':
            # Process configuration updates
            data = request.get_json()
            session_id = data.get('session_id')
            action = data.get('action')  # 'update_data_source', 'update_model', 'update_hyperparameters', 'next_step'
            
            if not session_id or session_id not in active_sessions:
                return jsonify({'status': 'error', 'error': 'Invalid session ID'}), 400
            
            session = active_sessions[session_id]
            
            # Handle different actions
            if action == 'update_data_source':
                data_source = data.get('data_source')  # 'nasa' or 'user'
                dataset = data.get('dataset')
                uploaded_file_info = data.get('uploaded_file_info')
                
                session.update({
                    'data_source': data_source,
                    'selected_dataset': dataset if data_source == 'nasa' else '',
                    'uploaded_file_info': uploaded_file_info if data_source == 'user' else None,
                    'step_progress': {**session.get('step_progress', {}), 'step1_complete': True}
                })
                
            elif action == 'update_model':
                model_type = data.get('model_type')
                
                session.update({
                    'selected_model': model_type,
                    'step_progress': {**session.get('step_progress', {}), 'step2_complete': True}
                })
                
            elif action == 'update_hyperparameters':
                hyperparameters = data.get('hyperparameters', {})
                preprocessing_config = data.get('preprocessing_config', {})
                
                session.update({
                    'hyperparameters': hyperparameters,
                    'preprocessing_config': preprocessing_config,
                    'step_progress': {**session.get('step_progress', {}), 'step3_complete': True},
                    'configured_at': datetime.now().isoformat()
                })
                
            elif action == 'next_step':
                current_step = session.get('current_step', 1)
                max_step = _calculate_max_step(session)
                
                if current_step < max_step:
                    session['current_step'] = current_step + 1
                    
            elif action == 'set_step':
                target_step = data.get('step', 1)
                max_step = _calculate_max_step(session)
                
                if 1 <= target_step <= max_step:
                    session['current_step'] = target_step
            
            # Update session timestamp
            session['updated_at'] = datetime.now().isoformat()
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'action': action,
                'session_info': session,
                'current_step': session.get('current_step', 1),
                'max_step': _calculate_max_step(session),
                'is_training_ready': _is_training_ready(session)
            })
            
    except Exception as e:
        logger.error(f"Error in select endpoint: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


def _calculate_max_step(session):
    """Calculate the maximum step user can reach based on current configuration."""
    data_source = session.get('data_source')
    selected_dataset = session.get('selected_dataset')
    uploaded_file_info = session.get('uploaded_file_info')
    
    if data_source == 'nasa' and selected_dataset:
        return 3
    elif data_source == 'user' and uploaded_file_info:
        return 3
    elif data_source in ['nasa', 'user']:
        return 2
    else:
        return 1


def _is_training_ready(session):
    """Check if all required configuration is complete for training."""
    selected_model = session.get('selected_model')
    current_step = session.get('current_step', 1)
    data_source = session.get('data_source')
    selected_dataset = session.get('selected_dataset')
    uploaded_file_info = session.get('uploaded_file_info')
    
    # Must have model selected
    if not selected_model:
        return False
        
    # Must be on step 3
    if current_step != 3:
        return False
    
    # Must have data source configured
    if data_source == 'nasa':
        return bool(selected_dataset)
    elif data_source == 'user':
        return bool(uploaded_file_info)
    
    return False


@app.route('/api/hyperparameters/<model_type>', methods=['GET'])
def get_model_hyperparameters(model_type):
    """Get hyperparameter configuration for a specific model type."""
    try:
        hyperparameters_config = {
            'linear_regression': [
                {'name': 'fit_intercept', 'label': 'Fit Intercept', 'type': 'select', 'options': ['true', 'false'], 'default': 'true'},
                {'name': 'normalize', 'label': 'Normalize', 'type': 'select', 'options': ['false', 'true'], 'default': 'false'}
            ],
            'svm': [
                {'name': 'C', 'label': 'C (Regularization)', 'type': 'number', 'default': 1.0, 'step': 0.1, 'min': 0.1},
                {'name': 'kernel', 'label': 'Kernel', 'type': 'select', 'options': ['rbf', 'linear', 'poly', 'sigmoid'], 'default': 'rbf'},
                {'name': 'gamma', 'label': 'Gamma', 'type': 'select', 'options': ['scale', 'auto'], 'default': 'scale'}
            ],
            'decision_tree': [
                {'name': 'max_depth', 'label': 'Max Depth', 'type': 'number', 'placeholder': 'None (unlimited)', 'min': 1},
                {'name': 'min_samples_split', 'label': 'Min Samples Split', 'type': 'number', 'default': 2, 'min': 2},
                {'name': 'min_samples_leaf', 'label': 'Min Samples Leaf', 'type': 'number', 'default': 1, 'min': 1},
                {'name': 'criterion', 'label': 'Criterion', 'type': 'select', 'options': ['gini', 'entropy'], 'default': 'gini'}
            ],
            'random_forest': [
                {'name': 'n_estimators', 'label': 'Number of Estimators', 'type': 'number', 'default': 100, 'min': 1},
                {'name': 'max_depth', 'label': 'Max Depth', 'type': 'number', 'placeholder': 'None (unlimited)', 'min': 1},
                {'name': 'min_samples_split', 'label': 'Min Samples Split', 'type': 'number', 'default': 2, 'min': 2},
                {'name': 'min_samples_leaf', 'label': 'Min Samples Leaf', 'type': 'number', 'default': 1, 'min': 1}
            ],
            'xgboost': [
                {'name': 'learning_rate', 'label': 'Learning Rate', 'type': 'number', 'default': 0.1, 'step': 0.01, 'min': 0.01, 'max': 1},
                {'name': 'max_depth', 'label': 'Max Depth', 'type': 'number', 'default': 6, 'min': 1},
                {'name': 'n_estimators', 'label': 'N Estimators', 'type': 'number', 'default': 100, 'min': 1},
                {'name': 'subsample', 'label': 'Subsample', 'type': 'number', 'default': 1.0, 'step': 0.1, 'min': 0.1, 'max': 1}
            ],
            'pca': [
                {'name': 'n_components', 'label': 'Number of Components', 'type': 'number', 'placeholder': 'Auto (min of features/samples)', 'min': 1},
                {'name': 'svd_solver', 'label': 'SVD Solver', 'type': 'select', 'options': ['auto', 'full', 'arpack', 'randomized'], 'default': 'auto'}
            ],
            'neural_network': [
                {'name': 'hidden_layer_sizes', 'label': 'Hidden Layer Sizes (comma-separated)', 'type': 'text', 'default': '100', 'placeholder': 'e.g., 100,50,25'},
                {'name': 'activation', 'label': 'Activation Function', 'type': 'select', 'options': ['relu', 'tanh', 'logistic'], 'default': 'relu'},
                {'name': 'learning_rate', 'label': 'Learning Rate', 'type': 'number', 'default': 0.001, 'step': 0.0001, 'min': 0.0001},
                {'name': 'max_iter', 'label': 'Max Iterations', 'type': 'number', 'default': 200, 'min': 1}
            ]
        }
        
        if model_type not in hyperparameters_config:
            return jsonify({'status': 'error', 'error': f'Unknown model type: {model_type}'}), 400
            
        return jsonify({
            'status': 'success',
            'model_type': model_type,
            'hyperparameters': hyperparameters_config[model_type]
        })
        
    except Exception as e:
        logger.error(f"Error getting hyperparameters for {model_type}: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Handle file uploads for user data."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'error': 'No file selected'}), 400
            
        if file and file.filename.endswith('.csv'):
            # Generate unique filename
            filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return jsonify({
                'status': 'success',
                'filename': filename,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'uploaded_at': datetime.now().isoformat()
            })
        else:
            return jsonify({'status': 'error', 'error': 'Only CSV files are allowed'}), 400
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/training', methods=['GET', 'POST'])
def training_progress():
    """
    Training endpoint - handles model training process and progress tracking.
    GET: Returns training progress for a session
    POST: Starts training process
    """
    try:
        if request.method == 'GET':
            session_id = request.args.get('session_id')
            
            if not session_id:
                return jsonify({'status': 'error', 'error': 'Session ID required'}), 400
            
            # Get training progress from ML API
            progress_info = training_api.get_training_progress(session_id)
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'progress': progress_info.get('progress', {}),
                'training_status': progress_info.get('status')
            })
            
        elif request.method == 'POST':
            # Start training process
            data = request.get_json()
            session_id = data.get('session_id')
            
            if not session_id or session_id not in active_sessions:
                return jsonify({'status': 'error', 'error': 'Invalid session ID'}), 400
            
            session_config = active_sessions[session_id]
            
            # Start training session in ML API
            training_api.start_training_session(session_id)
            
            # Prepare data loading configuration
            data_config = {
                'datasets': [session_config.get('dataset')] if session_config.get('dataset') else [],
                'filepath': session_config.get('uploaded_file'),
                'target_column': None  # Will be auto-detected
            }
            
            # Load data
            load_result = training_api.load_data_for_training(
                session_id=session_id,
                data_source=session_config['data_source'],
                data_config=data_config
            )
            
            if load_result['status'] != 'success':
                return jsonify(load_result), 400
            
            # Configure training
            training_config = {
                'model_type': session_config['model_type'],
                'target_column': None,  # Auto-detect
                'hyperparameters': session_config.get('hyperparameters', {}),
                'preprocessing_config': session_config.get('preprocessing_config', {})
            }
            
            config_result = training_api.configure_training(session_id, training_config)
            
            if config_result['status'] != 'success':
                return jsonify(config_result), 400
            
            # Start actual training (this will run in background)
            training_result = training_api.start_training(session_id)
            
            # Store training session info
            training_sessions[session_id] = {
                'started_at': datetime.now().isoformat(),
                'config': session_config,
                'status': 'training'
            }
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'training_started': True,
                'initial_progress': training_result
            })
            
    except Exception as e:
        logger.error(f"Error in training endpoint: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict_data():
    """
    Predict endpoint - handles data upload for prediction and returns prediction interface info.
    GET: Returns prediction interface info
    POST: Upload test data and prepare for prediction
    """
    try:
        if request.method == 'GET':
            session_id = request.args.get('session_id')
            
            if not session_id:
                return jsonify({'status': 'error', 'error': 'Session ID required'}), 400
            
            # Check if training is completed
            session_info = training_api.get_session_info(session_id)
            
            if session_info['status'] != 'success':
                return jsonify({'status': 'error', 'error': 'Training session not found'}), 404
            
            training_status = session_info['session_info'].get('status')
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'training_completed': training_status == 'completed',
                'training_status': training_status,
                'model_info': session_info['session_info'].get('model', {}),
                'ready_for_prediction': training_status == 'completed'
            })
            
        elif request.method == 'POST':
            # Handle test data upload or selection
            data = request.get_json()
            session_id = data.get('session_id')
            test_data_config = data.get('test_data_config', {})
            
            if not session_id:
                return jsonify({'status': 'error', 'error': 'Session ID required'}), 400
            
            # Store test data configuration for later use in results
            if session_id not in training_sessions:
                training_sessions[session_id] = {}
                
            training_sessions[session_id]['test_data_config'] = test_data_config
            training_sessions[session_id]['prediction_ready'] = True
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'test_data_configured': True,
                'test_config': test_data_config,
                'ready_for_results': True
            })
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv_file():
    """Upload CSV file for user data training."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'error': 'No file provided'}), 400
            
        file = request.files['file']
        session_id = request.form.get('session_id')
        
        if file.filename == '':
            return jsonify({'status': 'error', 'error': 'No file selected'}), 400
            
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'status': 'error', 'error': 'Only CSV files are allowed'}), 400
            
        # Validate session
        if session_id and session_id not in active_sessions:
            return jsonify({'status': 'error', 'error': 'Invalid session ID'}), 400
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(uploads_dir, filename)
        
        # Save file
        file.save(file_path)
        
        # Basic validation - try to read the CSV
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=5)  # Just read first 5 rows for validation
            num_rows = len(pd.read_csv(file_path))
            num_columns = len(df.columns)
            column_names = df.columns.tolist()
        except Exception as csv_error:
            # Remove the invalid file
            os.remove(file_path)
            return jsonify({
                'status': 'error',
                'error': f'Invalid CSV file: {str(csv_error)}'
            }), 400
        
        file_info = {
            'original_name': file.filename,
            'saved_name': filename,
            'file_path': file_path,
            'size': os.path.getsize(file_path),
            'uploaded_at': datetime.now().isoformat(),
            'num_rows': num_rows,
            'num_columns': num_columns,
            'columns': column_names
        }
        
        # Update session if provided
        if session_id and session_id in active_sessions:
            active_sessions[session_id].update({
                'uploaded_file_info': file_info,
                'data_source': 'user'
            })
        
        return jsonify({
            'status': 'success',
            'file_info': file_info,
            'message': f'File uploaded successfully. {num_rows} rows, {num_columns} columns detected.'
        })
        
    except Exception as e:
        logger.error(f"Error uploading CSV file: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/result', methods=['GET', 'POST'])
def prediction_results():
    """
    Result endpoint - returns prediction results, explanations, and visualizations.
    GET: Returns prediction results and model explanations
    POST: Generate explanations for specific instances
    """
    try:
        if request.method == 'GET':
            session_id = request.args.get('session_id')
            
            if not session_id:
                return jsonify({'status': 'error', 'error': 'Session ID required'}), 400
            
            # Check if we have a trained model
            session_info = training_api.get_session_info(session_id)
            
            if session_info['status'] != 'success':
                return jsonify({'status': 'error', 'error': 'Session not found'}), 404
            
            # Save the model with a unique name for prediction API
            model_name = f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_result = training_api.save_trained_model(session_id, model_name)
            
            if save_result['status'] == 'success':
                # Load model into prediction API
                model_path = save_result['model_path']
                load_result = prediction_api.load_model(model_path, session_id)
                
                if load_result['status'] == 'success':
                    # Generate sample predictions (mock data for now)
                    sample_results = {
                        'predictions': [
                            {
                                'id': 1,
                                'name': f'Object_{i+1}',
                                'prediction': 'Exoplanet' if i % 3 != 2 else 'False Positive',
                                'confidence': 0.85 + (i % 5) * 0.03,
                                'probabilities': [0.15 - (i % 5) * 0.03, 0.85 + (i % 5) * 0.03],
                                'class_probabilities': {
                                    'False Positive': 0.15 - (i % 5) * 0.03,
                                    'Exoplanet': 0.85 + (i % 5) * 0.03
                                }
                            }
                            for i in range(5)
                        ]
                    }
                    
                    # Calculate summary statistics
                    total_predictions = len(sample_results['predictions'])
                    exoplanet_count = len([p for p in sample_results['predictions'] if p['prediction'] == 'Exoplanet'])
                    false_positive_count = total_predictions - exoplanet_count
                    avg_confidence = sum([p['confidence'] for p in sample_results['predictions']]) / total_predictions
                    
                    return jsonify({
                        'status': 'success',
                        'session_id': session_id,
                        'model_info': load_result['model_info'],
                        'predictions': sample_results['predictions'],
                        'summary': {
                            'total_predictions': total_predictions,
                            'exoplanet_count': exoplanet_count,
                            'false_positive_count': false_positive_count,
                            'average_confidence': round(avg_confidence, 3)
                        },
                        'model_path': model_path
                    })
                else:
                    return jsonify({'status': 'error', 'error': 'Failed to load model for predictions'}), 500
            else:
                return jsonify({'status': 'error', 'error': 'Failed to save trained model'}), 500
            
        elif request.method == 'POST':
            # Generate explanation for specific instance
            data = request.get_json()
            session_id = data.get('session_id')
            instance_data = data.get('instance_data', {})
            explanation_type = data.get('explanation_type', 'local')  # 'local' or 'global'
            
            if not session_id:
                return jsonify({'status': 'error', 'error': 'Session ID required'}), 400
            
            # Generate explanation using explanation API
            if explanation_type == 'local':
                explanation_result = explanation_api.explain_prediction_local(
                    model_id=session_id,
                    instance_data=instance_data
                )
            else:
                # For global explanations, we'd need training data
                # This is a simplified version
                explanation_result = {
                    'status': 'success',
                    'explanation_type': 'global',
                    'message': 'Global explanation would require training data access'
                }
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'explanation': explanation_result
            })
            
    except Exception as e:
        logger.error(f"Error in result endpoint: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# Additional API endpoints for enhanced functionality

@app.route('/api/sessions/<session_id>', methods=['GET', 'DELETE'])
def manage_session(session_id):
    """Get or delete a specific session."""
    try:
        if request.method == 'GET':
            if session_id in active_sessions:
                session_info = active_sessions[session_id]
                training_info = training_sessions.get(session_id, {})
                
                return jsonify({
                    'status': 'success',
                    'session_id': session_id,
                    'session_info': session_info,
                    'training_info': training_info
                })
            else:
                return jsonify({'status': 'error', 'error': 'Session not found'}), 404
                
        elif request.method == 'DELETE':
            # Clean up session
            if session_id in active_sessions:
                del active_sessions[session_id]
            if session_id in training_sessions:
                del training_sessions[session_id]
            
            # Unload model if loaded
            prediction_api.unload_model(session_id)
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'message': 'Session deleted'
            })
            
    except Exception as e:
        logger.error(f"Error managing session {session_id}: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(active_sessions),
        'training_sessions': len(training_sessions),
        'loaded_models': len(prediction_api.loaded_models) if hasattr(prediction_api, 'loaded_models') else 0
    })


@app.route('/api/frontend-status', methods=['GET'])
def frontend_status():
    """Check frontend build status."""
    frontend_dist = Path(__file__).parent.parent / 'frontend' / 'dist'
    frontend_ready = frontend_dist.exists() and (frontend_dist / 'index.html').exists()
    
    return jsonify({
        'status': 'success',
        'frontend_built': frontend_ready,
        'dist_path': str(frontend_dist),
        'build_exists': frontend_dist.exists(),
        'index_exists': (frontend_dist / 'index.html').exists() if frontend_dist.exists() else False,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'status': 'error', 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'status': 'error', 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
