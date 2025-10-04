from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
import os
import time
import threading
import uuid

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
            dataset_path = None
            
            # Check if user uploaded a file
            if 'csv_file' in request.files and request.files['csv_file'].filename != '':
                file = request.files['csv_file']
                if file.filename.endswith('.csv'):
                    filename = f"user_upload_{file.filename}"
                    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(dataset_path)
                    dataset_source = 'user_upload'
                else:
                    flash('Please upload a valid CSV file', 'error')
                    return redirect(url_for('select'))
            
            # Check if user selected NASA dataset
            elif 'nasa_dataset' in request.form:
                nasa_dataset = request.form['nasa_dataset']
                dataset_source = 'nasa'
                dataset_path = f"data/{nasa_dataset}_raw.csv"
            
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
                'created_at': time.time()
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
        
        return render_template('predict.html')
    
    elif request.method == 'POST':
        # Handle prediction form submission
        try:
            # Get prediction dataset choice
            prediction_dataset_source = None
            prediction_dataset_path = None
            
            # Check if user uploaded a prediction file
            if 'prediction_csv_file' in request.files and request.files['prediction_csv_file'].filename != '':
                file = request.files['prediction_csv_file']
                if file.filename.endswith('.csv'):
                    filename = f"prediction_upload_{file.filename}"
                    prediction_dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(prediction_dataset_path)
                    prediction_dataset_source = 'user_upload'
                else:
                    flash('Please upload a valid CSV file for predictions', 'error')
                    return redirect(url_for('predict'))
            
            # Check if user selected NASA dataset for prediction
            elif 'nasa_prediction_dataset' in request.form:
                nasa_dataset = request.form['nasa_prediction_dataset']
                prediction_dataset_source = 'nasa'
                prediction_dataset_path = f"data/{nasa_dataset}_raw.csv"
            
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
                'prediction_dataset_path': prediction_dataset_path,
                'prediction_mode': prediction_mode,
                'output_format': output_format,
                'original_config': training_jobs[job_id]['config']
            }
            
            # Generate prediction job ID and store configuration
            prediction_job_id = str(uuid.uuid4())
            session['prediction_job_id'] = prediction_job_id
            
            # Store prediction job (in real app, this would trigger actual predictions)
            training_jobs[prediction_job_id] = {
                'config': prediction_config,
                'status': 'prediction_ready',
                'progress': 100,
                'created_at': time.time(),
                'type': 'prediction'
            }
            
            # In a real application, you would:
            # 1. Load the trained model from the training job
            # 2. Load and preprocess the prediction dataset
            # 3. Make predictions using the trained model
            # 4. Format results according to output_format
            # 5. Store results for display in /result
            
            flash(f'Prediction configuration saved. Ready to show results.', 'info')
            
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
    
    # Get prediction configuration
    prediction_config = training_jobs[prediction_job_id]['config']
    original_config = prediction_config['original_config']
    
    # Generate fake prediction data for display
    fake_prediction_data = [
        {
            'id': 1,
            'period': '3.52',
            'duration': '6.2h',
            'depth': '1200',
            'prediction': 'Exoplanet',
            'confidence': 94.2
        },
        {
            'id': 2,
            'period': '89.5',
            'duration': '13.1h',
            'depth': '890',
            'prediction': 'Exoplanet',
            'confidence': 87.6
        },
        {
            'id': 3,
            'period': '15.7',
            'duration': '4.8h',
            'depth': '2100',
            'prediction': 'False Positive',
            'confidence': 76.3
        },
        {
            'id': 4,
            'period': '7.2',
            'duration': '8.9h',
            'depth': '750',
            'prediction': 'Exoplanet',
            'confidence': 91.8
        },
        {
            'id': 5,
            'period': '124.3',
            'duration': '11.6h',
            'depth': '1450',
            'prediction': 'Exoplanet',
            'confidence': 88.4
        }
    ]
    
    # Calculate statistics
    exoplanets_count = sum(1 for item in fake_prediction_data if item['prediction'] == 'Exoplanet')
    false_positives_count = len(fake_prediction_data) - exoplanets_count
    overall_confidence = round(sum(item['confidence'] for item in fake_prediction_data) / len(fake_prediction_data), 1)
    
    # Model name mapping for display
    model_names = {
        'linear_regression': 'Linear Regression',
        'svm': 'Support Vector Machine',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'pca': 'Principal Component Analysis',
        'neural_network': 'Neural Network'
    }
    
    # Dataset name mapping
    dataset_names = {
        'kepler': 'Kepler Space Telescope Dataset',
        'k2': 'K2 Mission Dataset',
        'tess': 'TESS Survey Dataset',
        'user_upload': 'User Uploaded Dataset'
    }
    
    model_info = {
        'type': model_names.get(original_config['model_type'], original_config['model_type'].title()),
        'dataset': dataset_names.get(prediction_config['prediction_dataset_source'], 'Custom Dataset')
    }
    
    return render_template('result.html',
                         prediction_data=fake_prediction_data,
                         model_info=model_info,
                         exoplanets_count=exoplanets_count,
                         false_positives_count=false_positives_count,
                         overall_confidence=overall_confidence)

@app.route('/learn')
def learn():
    """Learn endpoint - for learning/educational content"""
    return jsonify({
        "message": "Learn endpoint",
        "status": "success",
        "description": "This endpoint will provide learning resources"
    })

def simulate_training(job_id, socketio):
    """Simulate training progress over 3 seconds"""
    if job_id not in training_jobs:
        return
    
    config = training_jobs[job_id]['config']
    model_type = config['model_type']
    dataset_source = config['dataset_source']
    
    # Model name mapping for display
    model_names = {
        'linear_regression': 'Linear Regression',
        'svm': 'Support Vector Machine',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'pca': 'Principal Component Analysis',
        'neural_network': 'Neural Network'
    }
    
    # Dataset name mapping
    dataset_names = {
        'kepler': 'Kepler Space Telescope Dataset',
        'k2': 'K2 Mission Dataset', 
        'tess': 'TESS Survey Dataset',
        'user_upload': 'User Uploaded Dataset'
    }
    
    model_name = model_names.get(model_type, model_type.title())
    dataset_name = dataset_names.get(dataset_source, 'Custom Dataset')
    
    # Send initial configuration
    socketio.emit('training_config', {
        'model_name': model_name,
        'dataset_name': dataset_name
    })
    
    # Training phases
    phases = [
        {'name': 'Loading dataset...', 'duration': 0.3},
        {'name': 'Preprocessing data...', 'duration': 0.5},
        {'name': 'Splitting train/test sets...', 'duration': 0.2},
        {'name': 'Initializing model...', 'duration': 0.3},
        {'name': 'Training model...', 'duration': 1.2},
        {'name': 'Validating performance...', 'duration': 0.4},
        {'name': 'Finalizing results...', 'duration': 0.1}
    ]
    
    total_duration = sum(phase['duration'] for phase in phases)
    elapsed_time = 0
    
    training_jobs[job_id]['status'] = 'running'
    
    for i, phase in enumerate(phases):
        if job_id not in training_jobs:
            break
            
        phase_progress = (elapsed_time / total_duration) * 100
        remaining_time = total_duration - elapsed_time
        
        socketio.emit('training_progress', {
            'percentage': phase_progress,
            'message': phase['name'],
            'eta': f'Estimated time remaining: {remaining_time:.1f}s',
            'phase': i + 1,
            'total_phases': len(phases)
        })
        
        # Simulate phase execution with gradual progress updates
        steps = max(10, int(phase['duration'] * 10))  # At least 10 steps per phase
        step_duration = phase['duration'] / steps
        
        for step in range(steps):
            if job_id not in training_jobs:
                break
                
            time.sleep(step_duration)
            elapsed_time += step_duration
            
            step_progress = (elapsed_time / total_duration) * 100
            remaining_time = max(0, total_duration - elapsed_time)
            
            socketio.emit('training_progress', {
                'percentage': min(99, step_progress),  # Cap at 99% until complete
                'message': phase['name'],
                'eta': f'Estimated time remaining: {remaining_time:.1f}s',
                'phase': i + 1,
                'total_phases': len(phases)
            })
    
    # Mark as completed
    if job_id in training_jobs:
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['progress'] = 100
        
        # Generate mock accuracy based on model type
        mock_accuracies = {
            'linear_regression': '85.3%',
            'svm': '89.7%', 
            'decision_tree': '82.1%',
            'random_forest': '91.2%',
            'xgboost': '93.8%',
            'pca': '78.5% variance explained',
            'neural_network': '90.4%'
        }
        
        accuracy = mock_accuracies.get(model_type, '87.6%')
        
        socketio.emit('training_complete', {
            'message': 'Training completed successfully!',
            'accuracy': accuracy,
            'model_type': model_name,
            'job_id': job_id
        })

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
    
    if training_jobs[job_id]['status'] != 'pending':
        emit('training_error', {'message': 'Training job already started or completed'})
        return
    
    # Start training simulation in background thread
    thread = threading.Thread(target=simulate_training, args=(job_id, socketio))
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)