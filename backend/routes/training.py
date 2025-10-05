"""
Training routes - Handle model training operations
"""
import threading
from flask import Blueprint, jsonify, request
from services import ml_service, session_service

training_bp = Blueprint('training', __name__)


@training_bp.route('/start', methods=['POST'])
def start_training():
    """Start a new training session"""
    try:
        config = request.get_json()
        
        if not config:
            return jsonify({'error': 'No configuration provided'}), 400
        
        # Create new session
        session_id = session_service.create_session()
        
        # Start training in background thread
        def train_model():
            try:
                # Update session status
                session_service.update_session(session_id, {
                    'status': 'initializing',
                    'progress': 10,
                    'current_step': 'Starting training session...',
                    'config': config
                })
                
                # Start ML training session
                ml_service.start_training_session(session_id)
                
                # Load data
                session_service.update_session(session_id, {
                    'progress': 20,
                    'current_step': 'Loading dataset...'
                })
                
                data_config = {}
                if config.get('dataset_source') == 'nasa':
                    data_config = {
                        'datasets': [config.get('dataset_name')],
                        'target_column': config.get('target_column', 'koi_disposition')
                    }
                    data_source = 'nasa'
                elif config.get('dataset_source') == 'upload':
                    data_config = {
                        'file_path': config.get('uploaded_file'),
                        'target_column': config.get('target_column')
                    }
                    data_source = 'custom'
                else:
                    raise ValueError('Invalid dataset source')
                
                ml_service.load_data_for_training(session_id, data_source, data_config)
                
                # Configure training
                session_service.update_session(session_id, {
                    'progress': 30,
                    'current_step': 'Configuring model...'
                })
                
                training_config = {
                    'model_type': config.get('model_type'),
                    'target_column': config.get('target_column', 'koi_disposition'),
                    'hyperparameters': config.get('hyperparameters', {})
                }
                
                ml_service.configure_training(session_id, training_config)
                
                # Start actual training
                session_service.update_session(session_id, {
                    'progress': 40,
                    'current_step': 'Training model...'
                })
                
                training_result = ml_service.start_training(session_id)
                
                # Save model
                session_service.update_session(session_id, {
                    'progress': 90,
                    'current_step': 'Saving model...'
                })
                
                model_name = config.get('model_name', f"{config.get('model_type')}_{config.get('dataset_name', 'custom')}_{session_id[:8]}")
                save_result = ml_service.save_trained_model(session_id, model_name)
                
                # Complete
                session_service.update_session(session_id, {
                    'status': 'completed',
                    'progress': 100,
                    'current_step': 'Training complete!',
                    'results': {
                        'training_result': training_result,
                        'model_path': save_result.get('model_path'),
                        'model_name': model_name
                    }
                })
                
            except Exception as e:
                session_service.update_session(session_id, {
                    'status': 'error',
                    'progress': -1,
                    'current_step': f'Error: {str(e)}',
                    'error': str(e)
                })
        
        # Start training thread
        training_thread = threading.Thread(target=train_model)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'message': 'Training started',
            'session_id': session_id,
            'status': 'started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/progress/<session_id>')
def get_training_progress(session_id):
    """Get training progress for a session"""
    try:
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get additional ML session info if available
        try:
            ml_session_info = ml_service.get_training_progress(session_id)
            if ml_session_info.get('status') == 'success':
                session_info.update({
                    'ml_session_info': ml_session_info.get('session_info', {})
                })
        except Exception:
            # ML session info not available yet
            pass
        
        return jsonify({
            'session_id': session_id,
            'session_info': session_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/status/<session_id>')
def get_training_status(session_id):
    """Get simplified training status"""
    try:
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'session_id': session_id,
            'status': session_info.get('status', 'unknown'),
            'progress': session_info.get('progress', 0),
            'current_step': session_info.get('current_step', 'Unknown'),
            'error': session_info.get('error')
        })
        
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