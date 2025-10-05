"""
Results routes - Handle prediction results and explanations
"""
import pandas as pd
from flask import Blueprint, jsonify, request
from services import ml_service, session_service

results_bp = Blueprint('results', __name__)


@results_bp.route('/predictions/<session_id>')
def get_session_results(session_id):
    """Get prediction results for a completed training session"""
    try:
        # Get session info
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        if session_info.get('status') != 'completed':
            return jsonify({'error': 'Training not completed yet'}), 400
        
        results = session_info.get('results', {})
        model_path = results.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'Model not available'}), 400
        
        # Load the trained model
        model_id = f"session_{session_id}"
        ml_service.load_model(model_path, model_id)
        
        # Get ML session info with data
        ml_session_info = ml_service.get_training_progress(session_id)
        
        if ml_session_info.get('status') != 'success':
            return jsonify({'error': 'Unable to get session data'}), 500
        
        session_data = ml_session_info.get('session_info', {})
        prepared_data = session_data.get('prepared_data')
        
        if not prepared_data:
            return jsonify({'error': 'Training data not available'}), 500
        
        # Make predictions on test set
        X_test = pd.DataFrame(prepared_data['X_test'])
        y_test = pd.Series(prepared_data['y_test'])
        
        # Get batch predictions
        test_data_list = X_test.to_dict('records')
        predictions_result = ml_service.predict_batch(model_id, test_data_list)
        
        if 'error' in predictions_result:
            return jsonify({'error': predictions_result['error']}), 500
        
        predictions = predictions_result.get('predictions', [])
        
        # Prepare results table
        results_table = []
        for i, (prediction_info, true_value) in enumerate(zip(predictions, y_test)):
            row = {
                'index': i,
                'true_label': true_value,
                'predicted_label': prediction_info.get('prediction'),
                'confidence': prediction_info.get('confidence', 0.0),
                'probabilities': prediction_info.get('probabilities', {}),
                'features': test_data_list[i]
            }
            results_table.append(row)
        
        # Calculate summary statistics
        correct_predictions = sum(1 for row in results_table if row['true_label'] == row['predicted_label'])
        accuracy = correct_predictions / len(results_table) if results_table else 0
        
        summary_stats = {
            'total_predictions': len(results_table),
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'model_metrics': results.get('training_result', {}).get('evaluation_metrics', {}),
            'training_metrics': results.get('training_result', {}).get('training_metrics', {})
        }
        
        return jsonify({
            'session_id': session_id,
            'model_name': results.get('model_name'),
            'results_table': results_table,
            'summary_stats': summary_stats,
            'feature_names': list(X_test.columns)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@results_bp.route('/explain/<session_id>')
def get_model_explanations(session_id):
    """Get model explanations for a trained model"""
    try:
        # Get session info
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        if session_info.get('status') != 'completed':
            return jsonify({'error': 'Training not completed yet'}), 400
        
        results = session_info.get('results', {})
        model_path = results.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'Model not available'}), 400
        
        # Load the trained model
        model_id = f"session_{session_id}"
        ml_service.load_model(model_path, model_id)
        
        # Get ML session info with data
        ml_session_info = ml_service.get_training_progress(session_id)
        session_data = ml_session_info.get('session_info', {})
        prepared_data = session_data.get('prepared_data')
        
        if not prepared_data:
            return jsonify({'error': 'Training data not available'}), 500
        
        # Get training and test data
        X_train = pd.DataFrame(prepared_data['X_train'])
        y_train = pd.Series(prepared_data['y_train'])
        X_test = pd.DataFrame(prepared_data['X_test'])
        y_test = pd.Series(prepared_data['y_test'])
        
        # Generate global explanations
        explanation_result = ml_service.explain_model_global(
            model_id, X_train, y_train, X_test, y_test
        )
        
        if 'error' in explanation_result:
            return jsonify({'error': explanation_result['error']}), 500
        
        return jsonify({
            'session_id': session_id,
            'model_name': results.get('model_name'),
            'explanations': explanation_result,
            'feature_names': list(X_train.columns)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@results_bp.route('/explain-prediction', methods=['POST'])
def explain_single_prediction():
    """Get explanation for a specific prediction instance"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        instance_index = data.get('instance_index')
        
        if not session_id or instance_index is None:
            return jsonify({'error': 'session_id and instance_index are required'}), 400
        
        # Get session info
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        results = session_info.get('results', {})
        model_path = results.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'Model not available'}), 400
        
        # Load the trained model
        model_id = f"session_{session_id}"
        ml_service.load_model(model_path, model_id)
        
        # Get ML session info with data
        ml_session_info = ml_service.get_training_progress(session_id)
        session_data = ml_session_info.get('session_info', {})
        prepared_data = session_data.get('prepared_data')
        
        if not prepared_data:
            return jsonify({'error': 'Training data not available'}), 500
        
        X_test = pd.DataFrame(prepared_data['X_test'])
        
        if instance_index >= len(X_test):
            return jsonify({'error': 'Instance index out of range'}), 400
        
        # Get specific instance
        instance_data = X_test.iloc[instance_index].to_dict()
        
        # Generate local explanation
        explanation_result = ml_service.explain_prediction_local(
            model_id, instance_data
        )
        
        if 'error' in explanation_result:
            return jsonify({'error': explanation_result['error']}), 500
        
        return jsonify({
            'session_id': session_id,
            'instance_index': instance_index,
            'instance_data': instance_data,
            'explanation': explanation_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@results_bp.route('/predict', methods=['POST'])
def make_custom_prediction():
    """Make prediction with custom input data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        input_features = data.get('features')
        
        if not session_id or not input_features:
            return jsonify({'error': 'session_id and features are required'}), 400
        
        # Get session info
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        results = session_info.get('results', {})
        model_path = results.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'Model not available'}), 400
        
        # Load the trained model
        model_id = f"session_{session_id}"
        ml_service.load_model(model_path, model_id)
        
        # Make prediction
        prediction_result = ml_service.predict_single(model_id, input_features)
        
        if 'error' in prediction_result:
            return jsonify({'error': prediction_result['error']}), 500
        
        # Get confidence if available
        confidence_result = ml_service.get_prediction_confidence(model_id, input_features)
        
        return jsonify({
            'session_id': session_id,
            'input_features': input_features,
            'prediction': prediction_result.get('prediction'),
            'confidence': prediction_result.get('confidence', 0.0),
            'probabilities': prediction_result.get('probabilities', {}),
            'confidence_analysis': confidence_result if 'error' not in confidence_result else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@results_bp.route('/download/<session_id>')
def download_results(session_id):
    """Download results as CSV or JSON"""
    try:
        format_type = request.args.get('format', 'json').lower()
        
        if format_type not in ['json', 'csv']:
            return jsonify({'error': 'Format must be json or csv'}), 400
        
        # Get session results
        session_info = session_service.get_session(session_id)
        
        if not session_info or session_info.get('status') != 'completed':
            return jsonify({'error': 'Session not found or not completed'}), 404
        
        # This would typically generate a file and return a download link
        # For now, return the results data
        results = session_info.get('results', {})
        
        return jsonify({
            'session_id': session_id,
            'download_format': format_type,
            'message': f'Results available for download in {format_type} format',
            'results_summary': {
                'model_name': results.get('model_name'),
                'training_completed': True,
                'model_path': results.get('model_path')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@results_bp.route('/visualization/<session_id>')
def get_visualization_data(session_id):
    """Get data for visualizations"""
    try:
        viz_type = request.args.get('type', 'feature_importance')
        
        # Get session info
        session_info = session_service.get_session(session_id)
        
        if not session_info:
            return jsonify({'error': 'Session not found'}), 404
        
        if session_info.get('status') != 'completed':
            return jsonify({'error': 'Training not completed yet'}), 400
        
        results = session_info.get('results', {})
        
        if viz_type == 'feature_importance':
            # Get feature importance data
            explanation_result = ml_service.explain_model_global(
                f"session_{session_id}", None, None, None, None
            )
            
            if 'error' not in explanation_result:
                return jsonify({
                    'session_id': session_id,
                    'visualization_type': viz_type,
                    'data': explanation_result.get('feature_importance', {})
                })
        
        elif viz_type == 'training_metrics':
            # Get training metrics
            training_metrics = results.get('training_result', {}).get('training_metrics', {})
            evaluation_metrics = results.get('training_result', {}).get('evaluation_metrics', {})
            
            return jsonify({
                'session_id': session_id,
                'visualization_type': viz_type,
                'data': {
                    'training_metrics': training_metrics,
                    'evaluation_metrics': evaluation_metrics
                }
            })
        
        else:
            return jsonify({'error': f'Unknown visualization type: {viz_type}'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500