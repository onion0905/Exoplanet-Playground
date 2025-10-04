from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
import os
import time
import threading
import uuid
import pandas as pd
import numpy as np

from ML.src.api.training_api import TrainingAPI

app = Flask(__name__)
app.secret_key = 'nasa_hackathon_secret_key_2024'  # For flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for training jobs
training_jobs = {}

# Initialize ML API
training_api = TrainingAPI()

# Dataset and model display mappings
MODEL_DISPLAY_NAMES = {
    'linear_regression': 'Linear Regression',
    'svm': 'Support Vector Machine',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'pca': 'Principal Component Analysis',
    'neural_network': 'Neural Network',
    'deep_learning': 'Neural Network'
}

DATASET_DISPLAY_NAMES = {
    'kepler': 'Kepler Space Telescope Dataset',
    'k2': 'K2 Mission Dataset',
    'tess': 'TESS Survey Dataset',
    'user': 'User Uploaded Dataset'
}

TARGET_COLUMN_MAP = {
    'kepler': 'koi_disposition',
    'k2': 'disposition',
    'tess': 'tfopwg_disp'
}

DISPLAY_FEATURE_MAP = {
    'kepler': ('koi_period', 'koi_duration', 'koi_depth'),
    'tess': ('pl_orbper', 'pl_trandurh', 'pl_trandep'),
    'k2': ('pl_orbper', 'pl_trandurh', 'pl_trandep')
}

PREDICTION_LABEL_MAP = {
    'planet': 'Exoplanet',
    'confirmed': 'Exoplanet',
    'confirmed_planet': 'Exoplanet',
    'kp': 'Exoplanet',
    'candidate': 'Candidate',
    'planet_candidate': 'Candidate',
    'pc': 'Candidate',
    'false_positive': 'False Positive',
    'fp': 'False Positive',
    'unknown': 'Unknown'
}


def get_model_display_name(model_type: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_type, model_type.replace('_', ' ').title())


def get_dataset_display_name(dataset_source: str, dataset_name: str = None) -> str:
    if dataset_source == 'nasa' and dataset_name:
        return DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name.title())
    return DATASET_DISPLAY_NAMES.get(dataset_source, dataset_name.title() if dataset_name else 'Custom Dataset')


def map_prediction_label(label: str) -> str:
    if label is None:
        return 'Unknown'
    normalized = str(label).strip().lower()
    return PREDICTION_LABEL_MAP.get(normalized, str(label).title())


def select_display_features(dataset_type: str, feature_columns: list) -> tuple:
    mapped = DISPLAY_FEATURE_MAP.get(dataset_type, ())
    selected = [col for col in mapped if col in feature_columns]

    for col in feature_columns:
        if col not in selected:
            selected.append(col)
        if len(selected) >= 3:
            break

    while len(selected) < 3:
        selected.append(None)

    return tuple(selected[:3])


def format_feature_value(value) -> str:
    if pd.isna(value):
        return 'N/A'
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def emit_training_progress(job_id: str, percentage: float, message: str,
                           phase: int = None, total_phases: int = None,
                           eta: str = '') -> None:
    training_jobs[job_id]['progress'] = percentage
    payload = {
        'percentage': percentage,
        'message': message,
        'eta': eta or '',
        'phase': phase,
        'total_phases': total_phases
    }
    socketio.emit('training_progress', payload)


def generate_prediction_results(training_job_id: str,
                                prediction_config: dict) -> dict:
    if training_job_id not in training_jobs:
        raise ValueError('Training job not found. Please train a model first.')

    training_job = training_jobs[training_job_id]
    if training_job.get('status') != 'completed':
        raise ValueError('Training job is not completed yet.')

    session_id = training_job.get('session_id') or training_job_id
    if session_id not in training_api.current_session:
        raise ValueError('Training session data not available. Please retrain your model.')

    session = training_api.current_session[session_id]
    model = session.get('model')
    if model is None or not getattr(model, 'is_trained', False):
        raise ValueError('Trained model not available for predictions.')

    feature_columns = session['training_config']['feature_columns']
    target_column = session['training_config']['target_column']

    dataset_source = prediction_config['prediction_dataset_source']
    dataset_name = prediction_config.get('prediction_dataset_name')
    dataset_path = prediction_config.get('prediction_dataset_path')

    if dataset_source == 'nasa':
        dataset_name = dataset_name or training_job['config'].get('dataset_name')
        if not dataset_name:
            raise ValueError('Please select a NASA dataset for predictions.')
        data_df = training_api.data_loader.load_nasa_dataset(dataset_name)
    elif dataset_source == 'user':
        if not dataset_path:
            raise ValueError('Please upload a CSV file for predictions.')
        data_df = training_api.data_loader.load_user_dataset(dataset_path)
    else:
        raise ValueError(f'Unknown prediction dataset source: {dataset_source}')

    missing_features = [col for col in feature_columns if col not in data_df.columns]
    if missing_features:
        missing_preview = ', '.join(missing_features[:5])
        raise ValueError(f'Missing required features in prediction data: {missing_preview}')

    prediction_features = data_df[feature_columns].copy()
    processor = training_api.data_processor
    prediction_features = processor.handle_missing_values(prediction_features, fit=False)
    prediction_features = processor.encode_categorical_features(prediction_features, fit=False)
    prediction_features = processor.scale_features(prediction_features, fit=False)

    prediction_features = prediction_features[feature_columns]

    if prediction_features.empty:
        raise ValueError('Prediction dataset is empty after preprocessing.')

    predictions = model.predict(prediction_features)
    if len(predictions) == 0:
        raise ValueError('No prediction results were generated for the provided data.')
    try:
        probabilities = model.predict_proba(prediction_features)
        max_probabilities = probabilities.max(axis=1)
    except Exception:
        probabilities = None
        max_probabilities = None

    dataset_type = training_job['config'].get('dataset_type')
    display_features = training_job.get('display_features') or select_display_features(dataset_type, feature_columns)
    display_columns = []
    for feature in display_features:
        if feature and feature in data_df.columns:
            display_columns.append(feature)
        else:
            display_columns.append(None)

    fallback_columns = feature_columns + [col for col in data_df.columns if col not in feature_columns]
    for idx, value in enumerate(display_columns):
        if value is None:
            for candidate in fallback_columns:
                if candidate not in display_columns and candidate != target_column:
                    display_columns[idx] = candidate
                    break

    display_labels = {
        'period': display_columns[0] or (feature_columns[0] if feature_columns else 'Feature 1'),
        'duration': display_columns[1] if len(display_columns) > 1 and display_columns[1] else (
            feature_columns[1] if len(feature_columns) > 1 else 'Feature 2'
        ),
        'depth': display_columns[2] if len(display_columns) > 2 and display_columns[2] else (
            feature_columns[2] if len(feature_columns) > 2 else 'Feature 3'
        )
    }

    records = []
    max_records = min(len(data_df), 25)
    for idx in range(max_records):
        record = {
            'id': idx + 1,
            'period': format_feature_value(data_df.iloc[idx][display_columns[0]]) if display_columns[0] else 'N/A',
            'duration': format_feature_value(data_df.iloc[idx][display_columns[1]]) if len(display_columns) > 1 and display_columns[1] else 'N/A',
            'depth': format_feature_value(data_df.iloc[idx][display_columns[2]]) if len(display_columns) > 2 and display_columns[2] else 'N/A',
            'prediction': map_prediction_label(predictions[idx]),
            'confidence': round(float(max_probabilities[idx]) * 100, 1) if max_probabilities is not None else 'N/A'
        }
        records.append(record)

    label_counts = {}
    for pred in predictions:
        label = map_prediction_label(pred)
        label_counts[label] = label_counts.get(label, 0) + 1

    exoplanets_count = label_counts.get('Exoplanet', 0)
    false_positives_count = label_counts.get('False Positive', 0)
    if max_probabilities is not None and len(max_probabilities) > 0:
        overall_confidence = round(float(np.mean(max_probabilities)) * 100, 1)
    else:
        overall_confidence = None

    results_df = data_df.copy()
    results_df['prediction'] = [map_prediction_label(pred) for pred in predictions]
    if max_probabilities is not None:
        results_df['confidence'] = (max_probabilities * 100).round(2)
        if model.target_classes and probabilities is not None:
            for class_index, class_name in enumerate(model.target_classes):
                column_name = f'prob_{class_name}'
                results_df[column_name] = probabilities[:, class_index]

    output_path = None
    if prediction_config.get('output_format') == 'csv':
        output_filename = f"prediction_results_{prediction_config['prediction_job_id']}.csv"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        results_df.to_csv(output_path, index=False)

    return {
        'records': records,
        'display_columns': display_labels,
        'summary': {
            'label_counts': label_counts,
            'exoplanets_count': exoplanets_count,
            'false_positives_count': false_positives_count,
            'overall_confidence': overall_confidence,
            'total_predictions': int(len(predictions))
        },
        'dataset_name': dataset_name,
        'dataset_source': dataset_source,
        'target_column': target_column,
        'output_file': output_path,
        'full_results': results_df
    }

@app.route('/')
def home():
    """Home endpoint - main page"""
    return render_template('index.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    """Select endpoint - for data or model selection"""
    if request.method == 'GET':
        return render_template('select.html')
    
    elif request.method == 'POST':
        # Handle form submission
        try:
            # Get dataset choice
            dataset_source = None
            dataset_name = None
            dataset_path = None
            
            # Check if user uploaded a file
            if 'csv_file' in request.files and request.files['csv_file'].filename != '':
                file = request.files['csv_file']
                if file.filename.endswith('.csv'):
                    filename = f"user_upload_{file.filename}"
                    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(dataset_path)
                    dataset_source = 'user'
                    dataset_name = 'user'
                else:
                    flash('Please upload a valid CSV file', 'error')
                    return redirect(url_for('select'))
            
            # Check if user selected NASA dataset
            elif 'nasa_dataset' in request.form:
                nasa_dataset = request.form['nasa_dataset']
                dataset_source = 'nasa'
                dataset_name = nasa_dataset
            
            else:
                flash('Please select a dataset or upload a CSV file', 'error')
                return redirect(url_for('select'))
            
            # Get model selection
            model_type = request.form.get('model')
            if not model_type:
                flash('Please select a machine learning model', 'error')
                return redirect(url_for('select'))
            
            # Collect hyperparameters based on model type
            hyperparameters = {}
            
            if model_type == 'linear_regression':
                hyperparameters = {
                    'fit_intercept': request.form.get('fit_intercept', 'true') == 'true',
                    'normalize': request.form.get('normalize', 'false') == 'true'
                }
            
            elif model_type == 'svm':
                hyperparameters = {
                    'C': float(request.form.get('C', 1.0)),
                    'kernel': request.form.get('kernel', 'rbf'),
                    'gamma': request.form.get('gamma', 'scale')
                }
            
            elif model_type == 'decision_tree':
                max_depth = request.form.get('max_depth')
                hyperparameters = {
                    'max_depth': int(max_depth) if max_depth else None,
                    'min_samples_split': int(request.form.get('min_samples_split', 2)),
                    'min_samples_leaf': int(request.form.get('min_samples_leaf', 1)),
                    'criterion': request.form.get('criterion', 'gini')
                }
            
            elif model_type == 'random_forest':
                max_depth = request.form.get('max_depth_rf')
                hyperparameters = {
                    'n_estimators': int(request.form.get('n_estimators', 100)),
                    'max_depth': int(max_depth) if max_depth else None,
                    'min_samples_split': int(request.form.get('min_samples_split_rf', 2)),
                    'min_samples_leaf': int(request.form.get('min_samples_leaf_rf', 1))
                }
            
            elif model_type == 'xgboost':
                hyperparameters = {
                    'learning_rate': float(request.form.get('learning_rate', 0.1)),
                    'max_depth': int(request.form.get('max_depth_xgb', 6)),
                    'n_estimators': int(request.form.get('n_estimators_xgb', 100)),
                    'subsample': float(request.form.get('subsample', 1.0))
                }
            
            elif model_type == 'pca':
                n_components = request.form.get('n_components')
                hyperparameters = {
                    'n_components': int(n_components) if n_components else None,
                    'svd_solver': request.form.get('svd_solver', 'auto')
                }
            
            elif model_type == 'neural_network':
                hidden_layers = request.form.get('hidden_layer_sizes', '100')
                # Parse comma-separated values
                try:
                    hidden_layer_sizes = tuple(map(int, hidden_layers.split(',')))
                except:
                    hidden_layer_sizes = (100,)
                
                hyperparameters = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': request.form.get('activation', 'relu'),
                    'learning_rate_init': float(request.form.get('learning_rate_nn', 0.001)),
                    'max_iter': int(request.form.get('max_iter', 200))
                }
            
            # Create training configuration
            training_config = {
                'dataset_source': dataset_source,
                'dataset_name': dataset_name,
                'dataset_path': dataset_path,
                'model_type': model_type,
                'hyperparameters': hyperparameters
            }
            
            # Generate unique job ID and store configuration
            job_id = str(uuid.uuid4())
            session['training_job_id'] = job_id
            training_jobs[job_id] = {
                'config': training_config,
                'status': 'pending',
                'progress': 0,
                'created_at': time.time(),
                'session_id': None,
                'training_metrics': None,
                'evaluation_metrics': None,
                'display_features': None
            }
            
            # Redirect to training page
            return redirect(url_for('training'))
            
        except Exception as e:
            flash(f'Error processing form: {str(e)}', 'error')
            return redirect(url_for('select'))

@app.route('/training')
def training():
    """Training endpoint - for model training operations"""
    # Check if we have a training job
    job_id = session.get('training_job_id')
    if not job_id or job_id not in training_jobs:
        flash('No training job found. Please start from model selection.', 'error')
        return redirect(url_for('select'))
    
    return render_template('training.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict endpoint - for making predictions"""
    if request.method == 'GET':
        # Check if we have a completed training job
        job_id = session.get('training_job_id')
        if not job_id or job_id not in training_jobs:
            flash('No training job found. Please start from model selection.', 'error')
            return redirect(url_for('select'))

        if training_jobs[job_id]['status'] != 'completed':
            flash('Model training not completed yet. Please wait for training to finish.', 'warning')
            return redirect(url_for('training'))

        job = training_jobs[job_id]
        evaluation_metrics = job.get('evaluation_metrics') or {}
        accuracy = evaluation_metrics.get('accuracy')
        if isinstance(accuracy, float) and accuracy <= 1:
            accuracy_display = f"{accuracy * 100:.2f}%"
        elif accuracy is not None:
            accuracy_display = str(accuracy)
        else:
            accuracy_display = 'N/A'

        model_details = {
            'type': get_model_display_name(job['config']['model_type']),
            'dataset': get_dataset_display_name(job['config'].get('dataset_source'), job['config'].get('dataset_name')),
            'accuracy': accuracy_display
        }

        return render_template('predict.html', model_details=model_details)

    elif request.method == 'POST':
        # Handle prediction form submission
        try:
            # Get prediction dataset choice
            prediction_dataset_source = None
            prediction_dataset_name = None
            prediction_dataset_path = None
            
            # Check if user uploaded a prediction file
            if 'prediction_csv_file' in request.files and request.files['prediction_csv_file'].filename != '':
                file = request.files['prediction_csv_file']
                if file.filename.endswith('.csv'):
                    filename = f"prediction_upload_{file.filename}"
                    prediction_dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(prediction_dataset_path)
                    prediction_dataset_source = 'user'
                    prediction_dataset_name = 'user'
                else:
                    flash('Please upload a valid CSV file for predictions', 'error')
                    return redirect(url_for('predict'))

            # Check if user selected NASA dataset for prediction
            elif 'nasa_prediction_dataset' in request.form:
                nasa_dataset = request.form['nasa_prediction_dataset']
                prediction_dataset_source = 'nasa'
                prediction_dataset_name = nasa_dataset

            else:
                flash('Please select a dataset for predictions or upload a CSV file', 'error')
                return redirect(url_for('predict'))
            
            # Get prediction options
            prediction_mode = request.form.get('prediction_mode', 'batch')
            output_format = request.form.get('output_format', 'detailed')
            
            # Get the original training job
            job_id = session.get('training_job_id')
            if not job_id or job_id not in training_jobs:
                flash('Training job not found. Please retrain your model.', 'error')
                return redirect(url_for('select'))
            
            # Create prediction configuration
            prediction_config = {
                'training_job_id': job_id,
                'prediction_dataset_source': prediction_dataset_source,
                'prediction_dataset_name': prediction_dataset_name,
                'prediction_dataset_path': prediction_dataset_path,
                'prediction_mode': prediction_mode,
                'output_format': output_format
            }

            # Generate prediction job ID and store configuration
            prediction_job_id = str(uuid.uuid4())
            session['prediction_job_id'] = prediction_job_id

            prediction_config['prediction_job_id'] = prediction_job_id

            prediction_results = generate_prediction_results(job_id, prediction_config)

            training_jobs[prediction_job_id] = {
                'config': prediction_config,
                'status': 'completed',
                'progress': 100,
                'created_at': time.time(),
                'type': 'prediction',
                'results': prediction_results
            }

            flash('Predictions generated successfully!', 'info')

            # Redirect to results page
            return redirect(url_for('result'))

        except Exception as e:
            flash(f'Error processing prediction request: {str(e)}', 'error')
            return redirect(url_for('predict'))

@app.route('/result')
def result():
    """Result endpoint - for displaying prediction results"""
    # Check if we have a prediction job
    prediction_job_id = session.get('prediction_job_id')
    if not prediction_job_id or prediction_job_id not in training_jobs:
        flash('No prediction results found. Please make predictions first.', 'error')
        return redirect(url_for('predict'))
    
    prediction_job = training_jobs[prediction_job_id]
    prediction_results = prediction_job.get('results')

    if not prediction_results:
        flash('Prediction results are not available. Please rerun the predictions.', 'error')
        return redirect(url_for('predict'))

    training_job_id = prediction_job['config']['training_job_id']
    training_job = training_jobs.get(training_job_id)
    if not training_job:
        flash('Original training job not found. Please retrain the model.', 'error')
        return redirect(url_for('select'))

    summary = prediction_results['summary']
    overall_confidence = summary.get('overall_confidence')
    overall_confidence_display = round(overall_confidence, 1) if overall_confidence is not None else 0

    model_info = {
        'type': get_model_display_name(training_job['config']['model_type']),
        'dataset': get_dataset_display_name(training_job['config'].get('dataset_source'),
                                            training_job['config'].get('dataset_name'))
    }

    return render_template(
        'result.html',
        prediction_data=prediction_results['records'],
        model_info=model_info,
        exoplanets_count=summary.get('exoplanets_count', 0),
        false_positives_count=summary.get('false_positives_count', 0),
        overall_confidence=overall_confidence_display,
        confidence_available=overall_confidence is not None,
        total_predictions=summary.get('total_predictions', 0),
        display_columns=prediction_results['display_columns'],
        label_counts=summary.get('label_counts', {}),
        download_path=prediction_results.get('output_file')
    )

@app.route('/learn')
def learn():
    """Learn endpoint - for learning/educational content"""
    return jsonify({
        "message": "Learn endpoint",
        "status": "success",
        "description": "This endpoint will provide learning resources"
    })

def run_training_job(job_id: str) -> None:
    """Run the full training pipeline using the ML training API."""
    if job_id not in training_jobs:
        socketio.emit('training_error', {'message': 'Training job not found'})
        return

    job = training_jobs[job_id]
    config = job['config']
    model_type = config['model_type']
    dataset_source = config['dataset_source']
    dataset_name = config.get('dataset_name')

    model_display = get_model_display_name(model_type)
    dataset_display = get_dataset_display_name(dataset_source, dataset_name)

    socketio.emit('training_config', {
        'model_name': model_display,
        'dataset_name': dataset_display
    })

    try:
        training_jobs[job_id]['status'] = 'running'
        training_jobs[job_id]['progress'] = 0

        emit_training_progress(job_id, 5, 'Initializing training session...', 1, 6)
        training_api.start_training_session(job_id)
        training_jobs[job_id]['session_id'] = job_id

        emit_training_progress(job_id, 15, 'Loading dataset...', 2, 6)
        if dataset_source == 'nasa':
            data_config = {'datasets': [dataset_name] if dataset_name else ['kepler']}
        elif dataset_source == 'user':
            if not config.get('dataset_path'):
                raise ValueError('No dataset file provided for user upload')
            data_config = {'filepath': config['dataset_path']}
        else:
            raise ValueError(f'Unknown dataset source: {dataset_source}')

        load_result = training_api.load_data_for_training(job_id, dataset_source, data_config)
        if load_result['status'] != 'success':
            raise ValueError(load_result.get('error', 'Failed to load dataset'))

        session = training_api.current_session[job_id]
        data = session['data']
        validation_info = load_result['data_info'].get('validation', {})
        dataset_type = validation_info.get('dataset_type') or dataset_name
        training_jobs[job_id]['config']['dataset_type'] = dataset_type

        target_column = config.get('target_column')
        if not target_column:
            if dataset_source == 'nasa' and dataset_name in TARGET_COLUMN_MAP:
                target_column = TARGET_COLUMN_MAP[dataset_name]
            elif dataset_type in TARGET_COLUMN_MAP:
                target_column = TARGET_COLUMN_MAP[dataset_type]
            else:
                for candidate in TARGET_COLUMN_MAP.values():
                    if candidate in data.columns:
                        target_column = candidate
                        break
            if not target_column:
                for candidate in ['label', 'target', 'class', 'outcome', 'status']:
                    if candidate in data.columns:
                        target_column = candidate
                        break
        if not target_column:
            raise ValueError('Unable to determine target column for dataset')

        recommended_features = validation_info.get('recommended_features') or []
        feature_columns = [col for col in recommended_features if col in data.columns]

        if not feature_columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]

        if not feature_columns:
            feature_columns = [col for col in data.columns if col != target_column]

        if not feature_columns:
            raise ValueError('No suitable feature columns found for training')

        config['target_column'] = target_column
        config['feature_columns'] = feature_columns

        display_features = select_display_features(dataset_type, feature_columns)
        training_jobs[job_id]['display_features'] = display_features

        emit_training_progress(job_id, 30, 'Configuring training pipeline...', 3, 6)
        configure_result = training_api.configure_training(job_id, {
            'model_type': model_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'hyperparameters': config.get('hyperparameters', {}),
            'preprocessing_config': {
                'handle_missing': True,
                'missing_strategy': 'median',
                'scale_features': True,
                'encode_categorical': True,
                'remove_outliers': False
            }
        })

        if configure_result['status'] != 'success':
            raise ValueError(configure_result.get('error', 'Failed to configure training'))

        emit_training_progress(job_id, 55, 'Training model...', 4, 6)
        training_result = training_api.start_training(job_id)
        if training_result['status'] != 'success':
            raise ValueError(training_result.get('error', 'Model training failed'))

        emit_training_progress(job_id, 85, 'Evaluating model performance...', 5, 6)
        training_jobs[job_id]['training_metrics'] = training_result.get('training_metrics')
        training_jobs[job_id]['evaluation_metrics'] = training_result.get('evaluation_metrics')

        evaluation_metrics = training_result.get('evaluation_metrics', {})
        accuracy = evaluation_metrics.get('accuracy')
        if isinstance(accuracy, float) and accuracy <= 1:
            accuracy_display = f"{accuracy * 100:.2f}%"
        elif accuracy is not None:
            accuracy_display = str(accuracy)
        else:
            accuracy_display = 'N/A'

        training_jobs[job_id]['status'] = 'completed'
        emit_training_progress(job_id, 100, 'Training completed!', 6, 6)

        socketio.emit('training_complete', {
            'message': 'Training completed successfully!',
            'accuracy': accuracy_display,
            'model_type': model_display,
            'job_id': job_id
        })

    except Exception as exc:
        training_jobs[job_id]['status'] = 'error'
        training_jobs[job_id]['error'] = str(exc)
        socketio.emit('training_error', {'message': str(exc)})

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print(f'Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_training')
def handle_start_training():
    """Start the training simulation when client requests it"""
    job_id = session.get('training_job_id')
    
    if not job_id or job_id not in training_jobs:
        emit('training_error', {'message': 'No valid training job found'})
        return
    
    if training_jobs[job_id]['status'] not in ['pending', 'error']:
        emit('training_error', {'message': 'Training job already started or completed'})
        return

    training_jobs[job_id]['status'] = 'pending'

    # Start training in background thread
    thread = threading.Thread(target=run_training_job, args=(job_id,))
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)