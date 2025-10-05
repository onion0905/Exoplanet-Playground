"""
Exoplanet Playground - Flask Backend API
Main application file for the web server.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
from pathlib import Path
import uuid
import logging
import json
import threading
import time
from typing import Dict, Any

# Add ML module to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.session_manager import SessionManager
from backend.utils.ml_integration import MLIntegration
from backend.utils.file_handler import FileHandler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = project_root / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

# Initialize utilities
session_manager = SessionManager()
ml_integration = MLIntegration()
file_handler = FileHandler(app.config['UPLOAD_FOLDER'])

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================
# BACKGROUND TASK FUNCTIONS
# ========================================

def _run_training_background(session_id: str, config: Dict[str, Any]):
    """Background task for model training."""
    try:
        # Start training
        result = ml_integration.start_custom_training(session_id, config)
        
        if result.get('status') == 'error':
            logger.error(f"Training start error: {result.get('error')}")
            session_manager.set_error(session_id, result.get('error', 'Training failed'))
            return
        
        # Poll progress until complete
        progress_info = None
        max_attempts = 300  # 150 seconds max
        attempts = 0
        
        while attempts < max_attempts:
            progress_info = ml_integration.get_training_progress(session_id)
            if progress_info:
                # Don't override status if already completed/error in session_manager
                current_session = session_manager.get_session(session_id)
                if current_session and current_session['status'] not in ['completed', 'error']:
                    session_manager.update_progress(
                        session_id,
                        progress_info['progress'],
                        progress_info['current_step'],
                        'running'  # Keep as running until we verify results
                    )
                
                if progress_info['status'] in ['completed', 'error']:
                    break
            
            time.sleep(0.5)
            attempts += 1
        
        # Check final status
        if not progress_info:
            session_manager.set_error(session_id, 'No progress information available')
            return
            
        if progress_info['status'] == 'error':
            session_manager.set_error(session_id, 'Training failed during execution')
            return
        
        # Get final results
        final_result = ml_integration.get_custom_training_result(session_id)
        if final_result:
            session_manager.set_result(session_id, final_result)
            logger.info(f"Training completed successfully for session {session_id}")
        else:
            logger.error(f"Failed to get training results for session {session_id}")
            session_manager.set_error(session_id, 'Failed to get training results')
        
    except Exception as e:
        logger.error(f"Error in background training: {str(e)}")
        import traceback
        traceback.print_exc()
        session_manager.set_error(session_id, str(e))


# ========================================
# CUSTOM TRAINING ENDPOINTS
# ========================================

@app.route('/api/custom/train', methods=['POST'])
def custom_train():
    """Start custom model training."""
    try:
        session_id = str(uuid.uuid4())
        
        # Parse request - handle both JSON and FormData
        if request.is_json:
            data = request.get_json()
            model_type = data.get('model_type')
            data_source = data.get('data_source')
        else:
            model_type = request.form.get('model_type')
            data_source = request.form.get('data_source')
        
        config = {
            'model_type': model_type,
            'data_source': data_source
        }
        
        # Handle data source
        if data_source == 'nasa':
            config['dataset_name'] = request.form.get('dataset_name') if not request.is_json else data.get('dataset_name')
        else:
            # Handle file uploads
            training_file = request.files.get('training_file')
            testing_file = request.files.get('testing_file')
            
            if training_file and testing_file:
                train_info = file_handler.save_uploaded_file(training_file, 'train')
                test_info = file_handler.save_uploaded_file(testing_file, 'test')
                
                config['train_data_path'] = train_info['filepath']
                config['test_data_path'] = test_info['filepath']
                config['data_format'] = request.form.get('data_format', 'kepler')
                config['dataset_name'] = config['data_format']  # Use format as dataset name
        
        # Parse hyperparameters
        if request.is_json:
            config['hyperparameters'] = data.get('hyperparameters', {})
        else:
            hp_str = request.form.get('hyperparameters')
            if hp_str:
                config['hyperparameters'] = json.loads(hp_str)
            else:
                config['hyperparameters'] = {}
        
        # Create session
        session_manager.create_session(session_id, 'custom_training', config)
        
        # Start training in background
        thread = threading.Thread(
            target=_run_training_background,
            args=(session_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Training started'
        }), 202
        
    except Exception as e:
        logger.error(f"Error in custom_train: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/custom/progress/<session_id>', methods=['GET'])
def custom_progress(session_id):
    """Get training progress for a session."""
    try:
        progress_info = session_manager.get_progress(session_id)
        
        if not progress_info:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            **progress_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error in custom_progress: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/custom/result/<session_id>', methods=['GET'])
def custom_result(session_id):
    """Get training results and predictions on test data."""
    try:
        # Check if session exists
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        # Check if completed
        if session['status'] == 'error':
            return jsonify({'success': False, 'error': session.get('error', 'Training failed')}), 500
        
        if session['status'] != 'completed':
            return jsonify({'success': False, 'error': 'Training not completed yet'}), 400
        
        # Get result
        result = session_manager.get_result(session_id)
        if not result:
            return jsonify({'success': False, 'error': 'Results not available'}), 500
        
        return jsonify({
            'success': True,
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error in custom_result: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ========================================
# PRETRAINED MODEL ENDPOINTS
# ========================================

@app.route('/api/pretrained/predict', methods=['POST'])
def pretrained_predict():
    """Run pretrained model prediction."""
    try:
        session_id = str(uuid.uuid4())
        
        # Parse request
        if request.is_json:
            data = request.get_json()
            data_source = data.get('data_source')
        else:
            data_source = request.form.get('data_source')
        
        config = {
            'data_source': data_source,
            'model_type': 'random_forest'  # Pretrained uses default
        }
        
        # Handle data source
        if data_source == 'nasa':
            config['dataset_name'] = request.form.get('dataset_name') if not request.is_json else data.get('dataset_name')
        else:
            # User uploaded data for fine-tuning
            training_file = request.files.get('training_file')
            if training_file:
                train_info = file_handler.save_uploaded_file(training_file, 'train')
                config['train_data_path'] = train_info['filepath']
                config['data_format'] = request.form.get('data_format', 'kepler')
                config['dataset_name'] = config['data_format']
        
        # Create session
        session_manager.create_session(session_id, 'pretrained_prediction', config)
        
        # Start prediction in background (reuse training logic)
        thread = threading.Thread(
            target=_run_training_background,
            args=(session_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Prediction started'
        }), 202
        
    except Exception as e:
        logger.error(f"Error in pretrained_predict: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/pretrained/progress/<session_id>', methods=['GET'])
def pretrained_progress(session_id):
    """Get prediction/fine-tuning progress."""
    return custom_progress(session_id)


@app.route('/api/pretrained/result/<session_id>', methods=['GET'])
def pretrained_result(session_id):
    """Get pretrained model results."""
    return custom_result(session_id)


# ========================================
# UTILITY ENDPOINTS
# ========================================

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List available NASA datasets."""
    return jsonify({
        'success': True,
        'datasets': ['kepler', 'tess', 'k2']
    }), 200


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available model types."""
    return jsonify({
        'success': True,
        'models': ['random_forest', 'decision_tree', 'xgboost', 'svm', 
                   'linear_regression', 'deep_learning', 'pca']
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'message': 'Exoplanet Playground API is running'
    }), 200


# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
