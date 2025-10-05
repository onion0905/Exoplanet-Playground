"""
Training routes - Handle model training operations
"""
import threading
import time
from flask import Blueprint, jsonify, request
from services import ml_service, session_service, data_service, results_service

training_bp = Blueprint('training', __name__)


@training_bp.route('/start', methods=['POST'])
def start_training():
    """Start a new training session with enhanced ML integration"""
    try:
        config = request.get_json()
        
        if not config:
            return jsonify({'error': 'No configuration provided'}), 400
        
        # Validate required fields
        required_fields = ['model_type', 'dataset_source', 'target_column']
        missing_fields = [field for field in required_fields if not config.get(field)]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Create new session
        session_id = session_service.create_session()
        
        # Start training in background thread
        def train_model():
            try:
                # Initialize training session
                session_service.update_session(session_id, {
                    'status': 'initializing',
                    'progress': 5,
                    'current_step': 'Initializing training session...',
                    'config': config
                })
                
                # Prepare data using our enhanced data service
                session_service.update_session(session_id, {
                    'progress': 10,
                    'current_step': 'Preparing training data...'
                })
                
                # Prepare data configuration for ML API
                ml_data_config = {
                    'target_column': config.get('target_column')
                }
                
                if config.get('dataset_source') == 'nasa':
                    ml_data_config['datasets'] = [config.get('dataset_name')]
                
                # Get data info from ML session
                ml_session = ml_service.training_api.current_session.get(session_id, {})
                ml_prepared_data = ml_session.get('prepared_data', {})
                
                if ml_prepared_data:
                    train_shape = ml_prepared_data['X_train'].shape
                    test_shape = ml_prepared_data['X_test'].shape
                    feature_count = train_shape[1]
                    target_classes = list(ml_prepared_data['y_train'].unique())
                else:
                    # Fallback values if ML session data not available
                    train_shape = (0, 0)
                    test_shape = (0, 0)
                    feature_count = 0
                    target_classes = []
                
                session_service.update_session(session_id, {
                    'progress': 25,
                    'current_step': f'Data loaded: {train_shape[0]} training samples, {test_shape[0]} test samples',
                    'data_info': {
                        'train_shape': list(train_shape),
                        'test_shape': list(test_shape),
                        'feature_count': feature_count,
                        'target_classes': target_classes
                    }
                })
                
                # Create and train model using ML service
                session_service.update_session(session_id, {
                    'progress': 35,
                    'current_step': f'Creating {config.get("model_type")} model...'
                })
                
                # Initialize ML training session
                ml_service.start_training_session(session_id)
                
                # Prepare ML data configuration
                ml_data_config = {
                    'datasets': [config.get('dataset_name')] if config.get('dataset_source') == 'nasa' else [],
                    'target_column': config.get('target_column')
                }
                
                # Load data into ML session using the ML system's three-class approach
                ml_data_result = ml_service.load_data_for_training(session_id, config.get('dataset_source'), ml_data_config)
                
                if ml_data_result.get('status') != 'success':
                    raise Exception(f"ML data loading failed: {ml_data_result.get('error', 'Unknown error')}")
                
                # Configure ML training
                session_service.update_session(session_id, {
                    'progress': 45,
                    'current_step': 'Configuring model parameters...'
                })
                
                training_config = {
                    'model_type': config.get('model_type'),
                    'target_column': config.get('target_column'),
                    'hyperparameters': config.get('hyperparameters', {})
                }
                
                ml_service.configure_training(session_id, training_config)
                
                # Start actual training
                session_service.update_session(session_id, {
                    'progress': 55,
                    'current_step': f'Training {config.get("model_type")} model...'
                })
                
                training_result = ml_service.start_training(session_id)
                
                if training_result.get('status') != 'success':
                    raise Exception(f"Training failed: {training_result.get('error', 'Unknown error')}")
                
                # Get the trained model
                trained_model = ml_service.training_api.current_session[session_id].get('model')
                if not trained_model:
                    raise Exception("Trained model not found in session")
                
                # Generate comprehensive testing results
                session_service.update_session(session_id, {
                    'progress': 80,
                    'current_step': 'Generating test predictions and analysis...'
                })
                
                # Get the ML session's prepared data for testing results
                ml_session = ml_service.training_api.current_session.get(session_id, {})
                ml_prepared_data = ml_session.get('prepared_data', {})
                
                if not ml_prepared_data:
                    raise Exception("No prepared data found in ML session")
                
                # Get target classes from the ML session's data
                target_classes = list(ml_prepared_data['y_test'].unique())
                feature_columns = list(ml_prepared_data['X_test'].columns)
                
                testing_results = results_service.generate_testing_results(
                    model=trained_model,
                    X_test=ml_prepared_data['X_test'],
                    y_test=ml_prepared_data['y_test'],
                    target_classes=target_classes,
                    feature_columns=feature_columns
                )
                
                # Save model
                session_service.update_session(session_id, {
                    'progress': 90,
                    'current_step': 'Saving trained model...'
                })
                
                model_name = config.get('model_name', 
                    f"{config.get('model_type')}_{config.get('dataset_name', 'custom')}_{int(time.time())}")
                
                save_result = ml_service.save_trained_model(session_id, model_name)
                
                # Complete training
                session_service.update_session(session_id, {
                    'status': 'completed',
                    'progress': 100,
                    'current_step': 'Training completed successfully!',
                    'ready_for_results': True,  # Signal frontend to redirect
                    'results': {
                        'model_name': model_name,
                        'model_path': save_result.get('model_path'),
                        'training_metrics': training_result.get('training_metrics', {}),
                        'evaluation_metrics': training_result.get('evaluation_metrics', {}),
                        'testing_results': testing_results,
                        'data_info': ml_session.get('data_info', {}),
                        'model_type': config.get('model_type'),
                        'dataset_source': config.get('dataset_source'),
                        'feature_columns': feature_columns,
                        'target_classes': target_classes
                    }
                })
                
                # Note: Trained model remains accessible via ML service session for predictions
                
            except Exception as e:
                session_service.update_session(session_id, {
                    'status': 'error',
                    'progress': -1,
                    'current_step': f'Training failed: {str(e)}',
                    'error': str(e),
                    'ready_for_results': False
                })
        
        # Start training thread
        training_thread = threading.Thread(target=train_model)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'message': 'Training started successfully',
            'session_id': session_id,
            'status': 'started'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500


@training_bp.route('/progress/<session_id>')
@training_bp.route('/progress')
def get_training_progress(session_id=None):
    """Get comprehensive training progress for a session"""
    try:
        # Get session_id from URL parameter or query parameter
        if not session_id:
            session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        # Prepare progress response
        progress_info = {
            'session_id': session_id,
            'status': session_info.get('status', 'unknown'),
            'progress': session_info.get('progress', 0),
            'current_step': session_info.get('current_step', 'Initializing...'),
            'ready_for_results': session_info.get('ready_for_results', False),
            'error': session_info.get('error'),
            'data_info': session_info.get('data_info', {}),
            'config': session_info.get('config', {}),
            'created_at': session_info.get('created_at')
        }
        
        # Add results summary if training is completed
        if session_info.get('status') == 'completed' and 'results' in session_info:
            results = session_info['results']
            progress_info['results_summary'] = {
                'model_name': results.get('model_name'),
                'model_type': results.get('model_type'),
                'accuracy': results.get('testing_results', {}).get('summary_metrics', {}).get('accuracy'),
                'total_samples': results.get('testing_results', {}).get('summary_metrics', {}).get('total_samples'),
                'feature_count': len(results.get('testing_results', {}).get('feature_columns', []))
            }
        
        return jsonify(progress_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/status/<session_id>')
def get_training_status(session_id):
    """Get simplified training status for quick polling"""
    try:
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        status_info = {
            'session_id': session_id,
            'status': session_info.get('status', 'unknown'),
            'progress': session_info.get('progress', 0),
            'current_step': session_info.get('current_step', 'Unknown'),
            'ready_for_results': session_info.get('ready_for_results', False),
            'error': session_info.get('error')
        }
        
        # Add completion time estimate based on progress
        if session_info.get('status') == 'initializing' and session_info.get('progress', 0) > 0:
            status_info['estimated_completion'] = 'Approximately 2-5 minutes remaining'
        elif session_info.get('progress', 0) > 50:
            status_info['estimated_completion'] = 'Almost complete'
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/cancel/<session_id>', methods=['POST'])
def cancel_training(session_id):
    """Cancel a training session"""
    try:
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        if session_info.get('status') in ['completed', 'error']:
            return jsonify({'message': 'Training already finished'}), 200
        
        # Update session to cancelled
        session_service.update_session(session_id, {
            'status': 'cancelled',
            'progress': -1,
            'current_step': 'Training cancelled by user'
        })
        
        return jsonify({
            'message': 'Training cancelled',
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/sessions')
def list_training_sessions():
    """List all training sessions"""
    try:
        sessions = session_service.list_sessions()
        
        # Filter out sensitive information
        filtered_sessions = {}
        for session_id, session_info in sessions.items():
            filtered_sessions[session_id] = {
                'id': session_info.get('id'),
                'created_at': session_info.get('created_at'),
                'status': session_info.get('status'),
                'progress': session_info.get('progress'),
                'current_step': session_info.get('current_step'),
                'model_name': session_info.get('results', {}).get('model_name')
            }
        
        return jsonify({
            'sessions': filtered_sessions,
            'count': len(filtered_sessions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/results/<session_id>')
@training_bp.route('/results')
def get_training_results(session_id=None):
    """Simple results endpoint for frontend"""
    try:
        # Get session_id
        if not session_id:
            session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        session_info = session_service.get_session(session_id)
        
        if not session_info or session_info.get('status') != 'completed':
            return jsonify({'error': 'Training not completed or session not found'}), 404
        
        results = session_info.get('results', {})
        testing_results = results.get('testing_results', {})
        
        # Get predictions table
        predictions_raw = testing_results.get('predictions_table', [])
        
        # Format predictions for frontend display
        predictions = []
        for i, pred in enumerate(predictions_raw[:20]):  # Show first 20 predictions
            pred_type = pred.get('predicted_label', 'unknown')
            
            # Map to display names
            if pred_type == 'planet':
                display_type = 'Confirmed Exoplanet'
            elif pred_type == 'candidate':
                display_type = 'Exoplanet Candidate'  
            elif pred_type == 'false_positive':
                display_type = 'False Positive'
            else:
                display_type = pred_type
            
            predictions.append({
                'id': i + 1,
                'name': f'Object {i + 1}',
                'predicted': display_type,
                'confidence': round(float(pred.get('confidence', 0)), 3),
                'actual': pred.get('true_label', 'Unknown'),
                'correct': pred.get('correct', False)
            })
        
        # Get metrics
        metrics = testing_results.get('summary_metrics', {})
        
        # Simple response for frontend
        return jsonify({
            'session_id': session_id,
            'accuracy': round(float(metrics.get('accuracy', 0)), 3),
            'total_predictions': len(predictions_raw),
            'predictions': predictions,
            'model_type': results.get('model_type', 'Random Forest'),
            'dataset': results.get('dataset_source', 'NASA Kepler')
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return jsonify({
            'error': str(e),
            'traceback': error_trace,
            'session_id': session_id,
            'debug_info': 'Error occurred while processing results'
        }), 500


@training_bp.route('/predict/<session_id>', methods=['POST'])
def make_prediction(session_id):
    """Make a single prediction using trained model"""
    try:
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        if session_info.get('status') != 'completed':
            return jsonify({'error': 'Training not completed yet'}), 400
        
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Features required for prediction'}), 400
        
        # Get trained model and feature info from session
        trained_model = session_info.get('trained_model_ref')
        feature_columns = session_info.get('feature_columns', [])
        target_classes = session_info.get('target_classes', [])
        
        if not trained_model:
            return jsonify({'error': 'Trained model not available in session'}), 500
        
        # Make prediction
        prediction_result = results_service.make_single_prediction(
            model=trained_model,
            input_features=data['features'],
            feature_columns=feature_columns,
            target_classes=target_classes
        )
        
        return jsonify({
            'session_id': session_id,
            'prediction_result': prediction_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/cleanup', methods=['POST'])
def cleanup_sessions():
    """Cleanup old training sessions"""
    try:
        max_age_hours = request.json.get('max_age_hours', 24) if request.json else 24
        
        session_service.cleanup_old_sessions(max_age_hours)
        
        return jsonify({
            'message': f'Cleaned up sessions older than {max_age_hours} hours'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500