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
    """Upload custom dataset for training - supports single file or separate train/test files"""
    try:
        upload_type = request.form.get('upload_type', 'single_file')  # 'single_file' or 'separate_files'
        
        uploaded_files = {}
        file_info = {}
        
        if upload_type == 'single_file':
            # Single file upload - will be split into train/test
            if 'data_file' not in request.files:
                return jsonify({'error': 'No data file provided'}), 400
            
            file = request.files['data_file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            uploaded_files['data_file'] = _process_uploaded_file(file, 'data')
            file_info = _analyze_uploaded_file(uploaded_files['data_file'])
            
        elif upload_type == 'separate_files':
            # Separate training and testing files
            required_files = ['training_file', 'testing_file']
            for file_key in required_files:
                if file_key not in request.files:
                    return jsonify({'error': f'Missing {file_key}'}), 400
                
                file = request.files[file_key]
                if file.filename == '':
                    return jsonify({'error': f'No {file_key} selected'}), 400
                
                uploaded_files[file_key] = _process_uploaded_file(file, file_key.split('_')[0])
            
            # Analyze both files
            train_info = _analyze_uploaded_file(uploaded_files['training_file'])
            test_info = _analyze_uploaded_file(uploaded_files['testing_file'])
            
            file_info = {
                'upload_type': 'separate_files',
                'training_file_info': train_info,
                'testing_file_info': test_info,
                'common_columns': list(set(train_info['column_names']) & set(test_info['column_names'])),
                'total_rows': train_info['rows'] + test_info['rows']
            }
        else:
            return jsonify({'error': 'Invalid upload_type. Use "single_file" or "separate_files"'}), 400
        
        # Store file information in response
        upload_info = {
            'upload_type': upload_type,
            'uploaded_files': uploaded_files,
            'file_analysis': file_info,
            'message': f'File(s) uploaded successfully - {upload_type} method'
        }
        
        return jsonify(upload_info)
        
    except Exception as e:
        # Clean up any uploaded files on error
        for file_path in uploaded_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        return jsonify({'error': str(e)}), 500


def _process_uploaded_file(file, file_type):
    """Process and save an uploaded file"""
    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise ValueError(f'File type not supported. Use: {", ".join(allowed_extensions)}')
    
    # Secure the filename and save
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{file_type}_{filename}"
    
    upload_dir = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, unique_filename)
    file.save(file_path)
    
    return unique_filename


def _analyze_uploaded_file(filename):
    """Analyze an uploaded file and return metadata"""
    upload_dir = current_app.config['UPLOAD_FOLDER']
    file_path = os.path.join(upload_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Uploaded file not found: {filename}')
    
    # Read file based on extension
    file_ext = os.path.splitext(filename)[1].lower()
    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Basic validation
        if df.empty:
            os.remove(file_path)
            raise ValueError('Uploaded file is empty')
        
        # Analyze numeric columns for potential features
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get dataset analysis
        dataset_info = {
            'filename': filename,
            'file_path': file_path,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(3).to_dict('records'),
            'potential_targets': categorical_columns[:5],  # First few categorical columns as potential targets
            'feature_candidates': numeric_columns[:20]  # First 20 numeric columns as potential features
        }
        
        return dataset_info
        
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise ValueError(f'Failed to process file: {str(e)}')


@select_bp.route('/validate-config', methods=['POST'])
def validate_training_config():
    """Validate training configuration before starting training"""
    try:
        config = request.get_json()
        
        if not config:
            return jsonify({'error': 'No configuration provided'}), 400
        
        # Required fields
        required_fields = ['model_type', 'dataset_source', 'target_column']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate model type
        from services import ml_service
        available_models = ml_service.get_available_models()
        if config['model_type'] not in available_models:
            validation_results['errors'].append(f'Invalid model type: {config["model_type"]}')
            validation_results['valid'] = False
        
        # Validate dataset source and specific requirements
        dataset_source = config['dataset_source']
        
        if dataset_source == 'nasa':
            if 'dataset_name' not in config:
                validation_results['errors'].append('Dataset name required for NASA datasets')
                validation_results['valid'] = False
            else:
                available_datasets = ml_service.get_available_datasets()
                if config['dataset_name'] not in available_datasets:
                    validation_results['errors'].append(f'Invalid dataset: {config["dataset_name"]}')
                    validation_results['valid'] = False
        
        elif dataset_source == 'upload':
            if 'uploaded_files' not in config:
                validation_results['errors'].append('Uploaded files configuration required for custom datasets')
                validation_results['valid'] = False
            else:
                uploaded_files = config['uploaded_files']
                upload_type = config.get('upload_type', 'single_file')
                
                if upload_type == 'single_file':
                    if 'data_file' not in uploaded_files:
                        validation_results['errors'].append('Data file required for single file upload')
                        validation_results['valid'] = False
                elif upload_type == 'separate_files':
                    required_files = ['training_file', 'testing_file']
                    for req_file in required_files:
                        if req_file not in uploaded_files:
                            validation_results['errors'].append(f'{req_file} required for separate files upload')
                            validation_results['valid'] = False
                else:
                    validation_results['errors'].append('Invalid upload_type. Use "single_file" or "separate_files"')
                    validation_results['valid'] = False
        else:
            validation_results['errors'].append('Invalid dataset source. Use "nasa" or "upload"')
            validation_results['valid'] = False
        
        # Validate target column
        target_column = config.get('target_column')
        if not target_column:
            validation_results['errors'].append('Target column is required')
            validation_results['valid'] = False
        
        # Validate hyperparameters if provided
        if 'hyperparameters' in config and config['hyperparameters']:
            try:
                valid_params = get_model_hyperparameters(config['model_type'])
                provided_params = list(config['hyperparameters'].keys())
                invalid_params = [param for param in provided_params if param not in valid_params]
                
                if invalid_params:
                    validation_results['warnings'].append(
                        f'Unknown hyperparameters will be ignored: {", ".join(invalid_params)}'
                    )
                
                # Validate hyperparameter values
                for param, value in config['hyperparameters'].items():
                    if param in valid_params:
                        if not _validate_hyperparameter_value(config['model_type'], param, value):
                            validation_results['warnings'].append(
                                f'Invalid value for hyperparameter {param}: {value}'
                            )
            except Exception as e:
                validation_results['warnings'].append(f'Could not validate hyperparameters: {str(e)}')
        
        # Validate test size if provided
        test_size = config.get('test_size', 0.2)
        if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
            validation_results['warnings'].append('test_size should be between 0 and 1, using default 0.2')
            config['test_size'] = 0.2
        
        # Add configuration recommendations
        recommendations = []
        model_type = config['model_type']
        
        if model_type == 'random_forest':
            recommendations.append('Random Forest works well with default parameters for most datasets')
        elif model_type == 'xgboost':
            recommendations.append('XGBoost may take longer to train but often provides better accuracy')
        elif model_type == 'deep_learning':
            recommendations.append('Deep Learning requires more data and training time but can capture complex patterns')
        
        # Prepare response
        response = {
            'validation_results': validation_results,
            'validated_config': config,
            'recommendations': recommendations
        }
        
        if validation_results['valid']:
            response['message'] = 'Configuration is valid and ready for training'
            return jsonify(response)
        else:
            response['message'] = 'Configuration validation failed'
            return jsonify(response), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _validate_hyperparameter_value(model_type, param, value):
    """Validate hyperparameter values based on model type and parameter"""
    try:
        # Basic validation rules for common hyperparameters
        validations = {
            'random_forest': {
                'n_estimators': lambda x: isinstance(x, int) and x > 0,
                'max_depth': lambda x: x is None or (isinstance(x, int) and x > 0),
                'min_samples_split': lambda x: isinstance(x, (int, float)) and x >= 2,
                'min_samples_leaf': lambda x: isinstance(x, (int, float)) and x >= 1
            },
            'xgboost': {
                'n_estimators': lambda x: isinstance(x, int) and x > 0,
                'learning_rate': lambda x: isinstance(x, (int, float)) and 0 < x <= 1,
                'max_depth': lambda x: isinstance(x, int) and x > 0,
                'subsample': lambda x: isinstance(x, (int, float)) and 0 < x <= 1
            },
            'svm': {
                'C': lambda x: isinstance(x, (int, float)) and x > 0,
                'gamma': lambda x: x == 'scale' or x == 'auto' or (isinstance(x, (int, float)) and x > 0)
            }
        }
        
        if model_type in validations and param in validations[model_type]:
            return validations[model_type][param](value)
        
        return True  # Allow unknown parameters to pass through
        
    except Exception:
        return False


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