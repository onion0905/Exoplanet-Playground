"""
Selection page routes - Dataset and model selection endpoints
"""
import os
import uuid
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
import pandas as pd

select_bp = Blueprint('select', __name__)


@select_bp.route('/datasets')
def get_available_datasets():
    """Get list of available NASA datasets"""
    try:
        from services import ml_service
        
        datasets = ml_service.get_available_datasets()
        
        # Add additional metadata for each dataset
        dataset_info = []
        for dataset_name in datasets:
            info = {
                'name': dataset_name,
                'display_name': get_dataset_display_name(dataset_name),
                'description': get_dataset_description(dataset_name),
                'features': get_dataset_features(dataset_name)
            }
            dataset_info.append(info)
        
        return jsonify({
            'datasets': dataset_info,
            'count': len(dataset_info)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@select_bp.route('/models')
def get_available_models():
    """Get list of available machine learning models"""
    try:
        from services import ml_service
        
        models = ml_service.get_available_models()
        
        # Add additional metadata for each model
        model_info = []
        for model_name in models:
            info = {
                'name': model_name,
                'display_name': get_model_display_name(model_name),
                'description': get_model_description(model_name),
                'hyperparameters': get_model_hyperparameters(model_name),
                'complexity': get_model_complexity(model_name),
                'training_time': get_model_training_time(model_name)
            }
            model_info.append(info)
        
        return jsonify({
            'models': model_info,
            'count': len(model_info)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@select_bp.route('/trained-models')
def get_trained_models():
    """Get list of already trained models"""
    try:
        from services import ml_service
        
        models = ml_service.get_trained_models()
        
        return jsonify({
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@select_bp.route('/upload', methods=['POST'])
def upload_custom_data():
    """Upload custom dataset for training"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type not supported. Use: {", ".join(allowed_extensions)}'}), 400
        
        # Secure the filename and save
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        upload_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, unique_filename)
        file.save(file_path)
        
        # Validate the uploaded data
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Basic validation
            if df.empty:
                os.remove(file_path)
                return jsonify({'error': 'Uploaded file is empty'}), 400
            
            # Get dataset info
            dataset_info = {
                'filename': unique_filename,
                'original_filename': filename,
                'file_path': file_path,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head().to_dict('records')
            }
            
            return jsonify({
                'message': 'File uploaded successfully',
                'dataset_info': dataset_info
            })
            
        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Failed to process file: {str(e)}'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@select_bp.route('/validate-config', methods=['POST'])
def validate_training_config():
    """Validate training configuration before starting training"""
    try:
        config = request.get_json()
        
        if not config:
            return jsonify({'error': 'No configuration provided'}), 400
        
        # Required fields
        required_fields = ['model_type', 'dataset_source']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Validate model type
        from services import ml_service
        available_models = ml_service.get_available_models()
        if config['model_type'] not in available_models:
            return jsonify({'error': f'Invalid model type: {config["model_type"]}'}), 400
        
        # Validate dataset source
        if config['dataset_source'] == 'nasa':
            if 'dataset_name' not in config:
                return jsonify({'error': 'Dataset name required for NASA datasets'}), 400
            
            available_datasets = ml_service.get_available_datasets()
            if config['dataset_name'] not in available_datasets:
                return jsonify({'error': f'Invalid dataset: {config["dataset_name"]}'}), 400
        
        elif config['dataset_source'] == 'upload':
            if 'uploaded_file' not in config:
                return jsonify({'error': 'Uploaded file path required for custom datasets'}), 400
        
        else:
            return jsonify({'error': 'Invalid dataset source. Use "nasa" or "upload"'}), 400
        
        # Validate hyperparameters if provided
        if 'hyperparameters' in config:
            valid_params = get_model_hyperparameters(config['model_type'])
            invalid_params = [param for param in config['hyperparameters'] if param not in valid_params]
            if invalid_params:
                return jsonify({
                    'warning': f'Unknown hyperparameters will be ignored: {", ".join(invalid_params)}',
                    'valid_parameters': valid_params
                })
        
        return jsonify({
            'message': 'Configuration is valid',
            'validated_config': config
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Helper functions for dataset and model metadata
def get_dataset_display_name(name):
    """Get human-readable dataset name"""
    names = {
        'kepler': 'Kepler Objects of Interest',
        'tess': 'TESS Objects of Interest', 
        'k2': 'K2 Mission Data'
    }
    return names.get(name, name.title())


def get_dataset_description(name):
    """Get dataset description"""
    descriptions = {
        'kepler': 'Data from NASA\'s Kepler Space Telescope mission focusing on exoplanet candidates',
        'tess': 'Transiting Exoplanet Survey Satellite data with stellar and planetary parameters',
        'k2': 'Extended mission data from Kepler\'s K2 campaign with additional exoplanet candidates'
    }
    return descriptions.get(name, 'NASA exoplanet dataset')


def get_dataset_features(name):
    """Get typical number of features for dataset"""
    features = {
        'kepler': 50,
        'tess': 35,
        'k2': 42
    }
    return features.get(name, 'Unknown')


def get_model_display_name(name):
    """Get human-readable model name"""
    names = {
        'random_forest': 'Random Forest',
        'decision_tree': 'Decision Tree',
        'linear_regression': 'Linear Regression',
        'svm': 'Support Vector Machine',
        'xgboost': 'XGBoost',
        'pca': 'Principal Component Analysis',
        'deep_learning': 'Deep Learning (Neural Network)'
    }
    return names.get(name, name.title().replace('_', ' '))


def get_model_description(name):
    """Get model description"""
    descriptions = {
        'random_forest': 'Ensemble method that combines multiple decision trees for robust predictions',
        'decision_tree': 'Tree-based model that makes decisions through a series of feature-based splits',
        'linear_regression': 'Linear model that finds the best linear relationship between features and target',
        'svm': 'Finds optimal boundary between classes using support vectors',
        'xgboost': 'Gradient boosting algorithm optimized for performance and accuracy',
        'pca': 'Dimensionality reduction technique that identifies principal components',
        'deep_learning': 'Multi-layer neural network capable of learning complex patterns'
    }
    return descriptions.get(name, 'Machine learning algorithm')


def get_model_hyperparameters(name):
    """Get typical hyperparameters for model"""
    hyperparameters = {
        'random_forest': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'decision_tree': ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion'],
        'linear_regression': ['fit_intercept', 'normalize'],
        'svm': ['C', 'kernel', 'gamma', 'degree'],
        'xgboost': ['n_estimators', 'learning_rate', 'max_depth', 'subsample'],
        'pca': ['n_components', 'whiten'],
        'deep_learning': ['hidden_layers', 'neurons_per_layer', 'learning_rate', 'batch_size', 'epochs']
    }
    return hyperparameters.get(name, [])


def get_model_complexity(name):
    """Get relative model complexity"""
    complexity = {
        'linear_regression': 'Low',
        'decision_tree': 'Low',
        'pca': 'Low',
        'random_forest': 'Medium',
        'svm': 'Medium',
        'xgboost': 'High',
        'deep_learning': 'High'
    }
    return complexity.get(name, 'Medium')


def get_model_training_time(name):
    """Get estimated training time"""
    times = {
        'linear_regression': '< 1 min',
        'decision_tree': '< 2 min',
        'pca': '< 1 min',
        'random_forest': '2-5 min',
        'svm': '3-8 min',
        'xgboost': '5-10 min',
        'deep_learning': '10-20 min'
    }
    return times.get(name, '5-10 min')