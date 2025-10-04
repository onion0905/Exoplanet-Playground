# NASA Exoplanet ML System - Complete API Documentation

This document provides comprehensive documentation for the NASA Exoplanet ML System APIs, including all parameters, return values, and usage examples.

## Table of Contents
1. [System Overview](#system-overview)
2. [User-Friendly API (ExoplanetMLAPI)](#user-friendly-api-exoplanetmlapı)
3. [Prediction API](#prediction-api)
4. [Explanation API](#explanation-api)
5. [Model Types and Hyperparameters](#model-types-and-hyperparameters)
6. [Datasets](#datasets)
7. [Complete Usage Examples](#complete-usage-examples)
8. [Error Handling](#error-handling)

## System Overview

The NASA Exoplanet ML System provides three main APIs for different use cases:

- **ExoplanetMLAPI**: High-level, user-friendly interface for training, prediction, and basic analysis
- **PredictionAPI**: Mid-level API for advanced prediction workflows and batch processing
- **ExplanationAPI**: Specialized API for model explainability and feature importance analysis

### Quick Start
```python
from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI

# Simple usage
api = ExoplanetMLAPI()

# Advanced usage with shared prediction API
prediction_api = PredictionAPI()
explanation_api = ExplanationAPI(prediction_api)
```

## User-Friendly API (ExoplanetMLAPI)

### Initialization
```python
api = ExoplanetMLAPI()
```

### Training Methods

#### `train_model()`
Train a new machine learning model on NASA exoplanet data.

**Parameters:**
- `model_type` (str, required): Type of model to train
  - Options: `'random_forest'`, `'decision_tree'`, `'linear_regression'`, `'svm'`, `'xgboost'`, `'pca'`, `'deep_learning'`
- `dataset_name` (str, required): NASA dataset to use for training
  - Options: `'kepler'`, `'tess'`, `'k2'`
- `model_name` (str, optional): Custom name for the trained model
  - Default: Auto-generated as `{model_type}_{dataset_name}_{timestamp}`
- `hyperparameters` (Dict[str, Any], optional): Custom hyperparameters for the model
  - Default: Uses model defaults

**Returns:**
```python
{
    'success': bool,
    'model_name': str,
    'model_type': str,
    'dataset': str,
    'training_time': float,
    'test_accuracy': float,
    'feature_count': int,
    'model_path': str,
    'metadata_path': str
}
```

**Example:**
```python
# Basic training
result = api.train_model(
    model_type='random_forest',
    dataset_name='kepler'
)

# Advanced training with custom parameters
result = api.train_model(
    model_type='random_forest',
    dataset_name='kepler',
    model_name='my_kepler_rf',
    hyperparameters={
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5
    }
)
```

#### `list_available_models()`
Get list of available model types.

**Parameters:** None

**Returns:**
```python
[
    'random_forest', 'decision_tree', 'linear_regression', 
    'svm', 'xgboost', 'pca', 'deep_learning'
]
```

#### `list_trained_models()`
Get information about all trained models.

**Parameters:** None

**Returns:**
```python
[
    {
        'model_name': str,
        'model_type': str,
        'dataset_name': str,
        'training_time': float,
        'test_accuracy': float,
        'feature_count': int,
        'trained_at': str,
        'model_path': str
    },
    # ... more models
]
```

### Prediction Methods

#### `predict_single()`
Make a prediction for a single exoplanet candidate.

**Parameters:**
- `model_name` (str, required): Name of the trained model to use
- `features` (Dict[str, Any], required): Feature values for prediction
  - Keys should match the feature names the model was trained on

**Returns:**
```python
{
    'success': bool,
    'prediction': str,           # Classification result
    'probability': float,        # Confidence probability (0-1)
    'confidence': str,          # 'High', 'Medium', 'Low'
    'model_used': str,
    'model_type': str,
    'dataset_trained_on': str,
    'error': str               # Only present if success=False
}
```

**Example:**
```python
# Kepler dataset features
kepler_features = {
    'koi_period': 365.25,
    'koi_prad': 1.0,
    'koi_teq': 288,
    'koi_insol': 1.0,
    'koi_model_snr': 15.5,
    'koi_steff': 5778,
    'koi_slogg': 4.44,
    'koi_srad': 1.0,
    # ... other features
}

result = api.predict_single('rf_kepler_123456', kepler_features)
```

#### `predict_batch()`
Make predictions for multiple exoplanet candidates.

**Parameters:**
- `model_name` (str, required): Name of the trained model to use
- `features_list` (List[Dict[str, Any]], required): List of feature dictionaries

**Returns:**
```python
[
    {
        'success': bool,
        'prediction': str,
        'probability': float,
        'confidence': str,
        'sample_id': int,
        'model_used': str,
        'model_type': str,
        'dataset_trained_on': str,
        'error': str               # Only present if success=False
    },
    # ... more predictions
]
```

### Analysis Methods

#### `get_feature_importance()`
Get feature importance scores for a trained model.

**Parameters:**
- `model_name` (str, required): Name of the trained model
- `top_n` (int, optional): Number of top features to return
  - Default: 10

**Returns:**
```python
{
    'success': bool,
    'model_name': str,
    'model_type': str,
    'top_features': Dict[str, float],    # Top N features with scores
    'all_features': Dict[str, float],    # All features with scores
    'total_features': int,
    'note': str,                         # Additional information
    'error': str                         # Only present if success=False
}
```

#### `get_sample_data()`
Get sample data from a NASA dataset for testing or analysis.

**Parameters:**
- `dataset_name` (str, required): Dataset to sample from (`'kepler'`, `'tess'`, `'k2'`)
- `n_samples` (int, optional): Number of samples to return
  - Default: 10

**Returns:**
```python
{
    'success': bool,
    'dataset_name': str,
    'sample_data': List[Dict[str, Any]], # List of feature dictionaries
    'sample_size': int,
    'feature_names': List[str],
    'error': str                         # Only present if success=False
}
```

## Prediction API

### Initialization
```python
prediction_api = PredictionAPI()
```

### Model Management

#### `load_model()`
Load a trained model for predictions.

**Parameters:**
- `model_path` (str, required): Path to the model file (`.joblib`)
- `model_id` (str, optional): Custom ID for the loaded model
  - Default: Uses filename stem

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'model_info': {
        'model_name': str,
        'is_trained': bool,
        'feature_names': List[str],
        'target_classes': List[str],
        'training_history': Dict[str, Any]
    },
    'metadata': Dict[str, Any],
    'error': str                         # Only present if status='error'
}
```

### Prediction Methods

#### `predict_single()`
Make a prediction for a single instance (low-level version).

**Parameters:**
- `model_id` (str, required): ID of the loaded model
- `input_data` (Dict[str, Any], required): Feature values

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'prediction': str,
    'probabilities': List[float],        # Raw probability array
    'class_probabilities': Dict[str, float], # Named class probabilities
    'input_data': Dict[str, Any],
    'error': str                         # Only present if status='error'
}
```

#### `predict_batch()`
Make predictions for multiple instances (low-level version).

**Parameters:**
- `model_id` (str, required): ID of the loaded model
- `input_data` (List[Dict[str, Any]], required): List of feature dictionaries

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'results': [
        {
            'index': int,
            'prediction': str,
            'probabilities': List[float],
            'class_probabilities': Dict[str, float],
            'input_data': Dict[str, Any]
        },
        # ... more results
    ],
    'batch_size': int,
    'error': str                         # Only present if status='error'
}
```

#### `predict_csv()`
Make predictions for data in a CSV file.

**Parameters:**
- `model_id` (str, required): ID of the loaded model
- `csv_path` (str, required): Path to the CSV file

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'predictions_made': int,
    'output_file': str,                  # Path to results CSV
    'input_file': str,
    'processing_time': float,
    'error': str                         # Only present if status='error'
}
```

## Explanation API

### Initialization
```python
# Standalone
explanation_api = ExplanationAPI()

# With shared prediction API (recommended)
prediction_api = PredictionAPI()
explanation_api = ExplanationAPI(prediction_api)
```

### Global Explanations

#### `explain_model_global()`
Generate comprehensive feature importance analysis for a model.

**Parameters:**
- `model_id` (str, required): ID of the loaded model
- `X_train` (pd.DataFrame, required): Training feature data
- `y_train` (pd.Series, required): Training target data
- `X_test` (pd.DataFrame, required): Test feature data
- `y_test` (pd.Series, required): Test target data
- `methods` (List[str], optional): Explanation methods to use
  - Default: `['model_importance', 'column_drop', 'average_replacement']`
  - Available: `'model_importance'`, `'column_drop'`, `'average_replacement'`, `'feature_ranking'`

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'explanation_type': 'global',
    'results': {
        'model_importance': Dict[str, float],    # Built-in feature importance
        'column_drop': Dict[str, Dict],          # Column drop analysis results
        'average_replacement': Dict[str, float], # Average replacement importance
        'feature_ranking': Dict[str, float]      # Ranked feature importance
    },
    'top_features': List[Tuple[str, float]],    # Top 10 features as (name, score) tuples
    'methods_used': List[str],
    'error': str                                 # Only present if status='error'
}
```

**Example:**
```python
# Load sample data
sample_data = api.get_sample_data('kepler', n_samples=100)
df = pd.DataFrame(sample_data['sample_data'])
X = df.drop(columns=['koi_disposition'])
y = df['koi_disposition']

# Split for explanation
X_train = X[:50]
X_test = X[50:]
y_train = y[:50]  
y_test = y[50:]

# Generate explanation
explanation = explanation_api.explain_model_global(
    model_id='rf_kepler_123456',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    methods=['model_importance', 'column_drop']
)
```

### Local Explanations

#### `explain_prediction_local()`
Generate explanation for a single prediction.

**Parameters:**
- `model_id` (str, required): ID of the loaded model
- `instance_data` (Dict[str, Any], required): Single instance to explain
- `X_train` (pd.DataFrame, optional): Training data for reference
- `y_train` (pd.Series, optional): Training targets for reference

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'explanation_type': 'local',
    'prediction_result': {
        'prediction': str,
        'probabilities': List[float],
        'class_probabilities': Dict[str, float]
    },
    'feature_contributions': Dict[str, float],   # Feature contributions to this prediction
    'instance_data': Dict[str, Any],
    'error': str                                 # Only present if status='error'
}
```

### Feature Analysis

#### `analyze_feature_importance_drop()`
Analyze feature importance using column dropping method.

**Parameters:**
- `model_id` (str, required): ID of the loaded model
- `X_train` (pd.DataFrame, required): Training feature data
- `y_train` (pd.Series, required): Training target data
- `X_test` (pd.DataFrame, required): Test feature data
- `y_test` (pd.Series, required): Test target data
- `features_to_analyze` (List[str], optional): Specific features to analyze
  - Default: Analyzes all features

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'method': 'column_drop_importance',
    'baseline_accuracy': float,
    'feature_importance_ranking': List[Tuple[str, Dict]],  # Sorted by importance
    'detailed_results': Dict[str, Dict],                   # Full results per feature
    'features_analyzed': int,
    'error': str                                           # Only present if status='error'
}
```

#### `compare_feature_importance_methods()`
Compare different feature importance methods on the same model.

**Parameters:**
- `model_id` (str, required): ID of the loaded model
- `X_train` (pd.DataFrame, required): Training feature data
- `y_train` (pd.Series, required): Training target data
- `X_test` (pd.DataFrame, required): Test feature data
- `y_test` (pd.Series, required): Test target data

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'error',
    'comparison_results': {
        'model_importance': Dict[str, float],
        'column_drop': Dict[str, float],
        'average_replacement': Dict[str, float]
    },
    'correlation_analysis': Dict[str, float],    # Correlation between methods
    'consensus_ranking': List[str],              # Features ranked by consensus
    'error': str                                 # Only present if status='error'
}
```

### Decision Path Analysis

#### `explain_model_decision_path()`
Explain decision path for tree-based models (Decision Trees, Random Forest).

**Parameters:**
- `model_id` (str, required): ID of the loaded model (must be tree-based)
- `instance_data` (Dict[str, Any], required): Instance to analyze

**Returns:**
```python
{
    'model_id': str,
    'status': 'success' | 'not_supported' | 'error',
    'explanation_type': 'decision_path',
    'instance_data': Dict[str, Any],
    'prediction': str,
    'tree_rules': str,                           # Human-readable tree rules
    'model_type': str,
    'message': str,                              # Status message
    'error': str                                 # Only present if status='error'
}
```

## Model Types and Hyperparameters

### Random Forest (`'random_forest'`)
**Default Hyperparameters:**
```python
{
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}
```

### Decision Tree (`'decision_tree'`)
**Default Hyperparameters:**
```python
{
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}
```

### Logistic Regression (`'linear_regression'`)
**Default Hyperparameters:**
```python
{
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs'
}
```

### Support Vector Machine (`'svm'`)
**Default Hyperparameters:**
```python
{
    'C': 1.0,
    'kernel': 'rbf',
    'random_state': 42
}
```

### XGBoost (`'xgboost'`)
**Default Hyperparameters:**
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}
```

### PCA + Logistic Regression (`'pca'`)
**Default Hyperparameters:**
```python
{
    'n_components': 0.95,  # Retain 95% of variance
    'random_state': 42
}
```

### Deep Learning (`'deep_learning'`)
**Default Hyperparameters:**
```python
{
    'hidden_layers': [64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32
}
```

## Datasets

### Kepler Dataset (`'kepler'`)
- **Target Column:** `koi_disposition`
- **Classes:** `'CONFIRMED'`, `'CANDIDATE'`, `'FALSE POSITIVE'`
- **Records:** ~9,564
- **Key Features:** `koi_period`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_model_snr`, `koi_steff`, `koi_slogg`, `koi_srad`

### TESS Dataset (`'tess'`)
- **Target Column:** `tfopwg_disp`
- **Classes:** `'FP'` (False Positive), `'PC'` (Planet Candidate), `'KP'` (Known Planet), `'APC'` (Astrophysical False Positive Candidate)
- **Records:** ~7,699
- **Key Features:** `st_tmag`, `pl_trandep`, `pl_orbper`, `pl_eqt`, `st_disterr2`

### K2 Dataset (`'k2'`)
- **Target Column:** `disposition`
- **Classes:** `'CONFIRMED'`, `'CANDIDATE'`
- **Records:** ~4,004
- **Key Features:** Similar to Kepler but with some differences in available measurements

## Complete Usage Examples

### Example 1: End-to-End Model Training and Prediction
```python
from ML.src.api.user_api import ExoplanetMLAPI

# Initialize API
api = ExoplanetMLAPI()

# Train a model
training_result = api.train_model(
    model_type='random_forest',
    dataset_name='kepler',
    model_name='my_kepler_classifier',
    hyperparameters={
        'n_estimators': 200,
        'max_depth': 15
    }
)

print(f"Training completed: {training_result['test_accuracy']:.3f} accuracy")

# Get sample data for prediction
sample_data = api.get_sample_data('kepler', n_samples=5)

# Make predictions
for i, sample in enumerate(sample_data['sample_data']):
    # Remove target column
    features = {k: v for k, v in sample.items() if k != 'koi_disposition'}
    
    # Predict
    prediction = api.predict_single('my_kepler_classifier', features)
    
    print(f"Sample {i+1}: {prediction['prediction']} "
          f"(confidence: {prediction['confidence']}, "
          f"probability: {prediction['probability']:.3f})")

# Analyze feature importance
importance = api.get_feature_importance('my_kepler_classifier', top_n=5)
print("\nTop 5 Important Features:")
for feature, score in importance['top_features'].items():
    print(f"  {feature}: {score:.4f}")
```

### Example 2: Advanced Explanation Analysis
```python
from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI
import pandas as pd

# Initialize APIs with shared prediction API
user_api = ExoplanetMLAPI()
prediction_api = PredictionAPI()
explanation_api = ExplanationAPI(prediction_api)

# Train model
result = user_api.train_model('random_forest', 'kepler', 'explanation_model')
model_name = result['model_name']

# Load model for explanation
model_path = f"ML/models/user/{model_name}.joblib"
load_result = prediction_api.load_model(model_path, model_name)

# Get data for explanation
sample_data = user_api.get_sample_data('kepler', n_samples=200)
df = pd.DataFrame(sample_data['sample_data'])

# Prepare data
X = df.drop(columns=['koi_disposition'])
y = df['koi_disposition']
X_train = X[:100]
X_test = X[100:]
y_train = y[:100]
y_test = y[100:]

# Generate comprehensive explanation
explanation = explanation_api.explain_model_global(
    model_id=model_name,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    methods=['model_importance', 'column_drop']
)

# Print results
if explanation['status'] == 'success':
    print("Top 5 Most Important Features:")
    for i, (feature, score) in enumerate(explanation['top_features'][:5], 1):
        print(f"  {i}. {feature}: {score:.4f}")
    
    # Analyze specific prediction
    sample_instance = df.iloc[0].drop('koi_disposition').to_dict()
    local_explanation = explanation_api.explain_prediction_local(
        model_id=model_name,
        instance_data=sample_instance
    )
    
    if local_explanation['status'] == 'success':
        pred_result = local_explanation['prediction_result']
        print(f"\nPrediction: {pred_result['prediction']}")
        print("Feature Contributions:")
        contributions = local_explanation['feature_contributions']
        for feature, contrib in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"  {feature}: {contrib:+.4f}")
```

### Example 3: Batch Processing
```python
from ML.src.api.prediction_api import PredictionAPI
import pandas as pd

# Initialize API
prediction_api = PredictionAPI()

# Load model
model_path = "ML/models/user/rf_kepler_123456.joblib"
load_result = prediction_api.load_model(model_path, "batch_model")

# Prepare batch data
batch_data = [
    {
        'koi_period': 365.25,
        'koi_prad': 1.0,
        'koi_teq': 288,
        # ... other features
    },
    {
        'koi_period': 10.5,
        'koi_prad': 2.1,
        'koi_teq': 800,
        # ... other features
    },
    # ... more instances
]

# Make batch predictions
batch_result = prediction_api.predict_batch("batch_model", batch_data)

if batch_result['status'] == 'success':
    print(f"Processed {batch_result['batch_size']} instances")
    for result in batch_result['results'][:5]:  # Show first 5
        print(f"Instance {result['index']}: {result['prediction']} "
              f"(max prob: {max(result['probabilities']):.3f})")

# Process CSV file
csv_result = prediction_api.predict_csv("batch_model", "data/new_candidates.csv")
if csv_result['status'] == 'success':
    print(f"Processed {csv_result['predictions_made']} predictions")
    print(f"Results saved to: {csv_result['output_file']}")
```

## Error Handling

All APIs use consistent error handling patterns:

### Success Responses
- Always include `status: 'success'` or `success: True`
- Contain expected data fields

### Error Responses  
- Include `status: 'error'` or `success: False`
- Contain `error` field with descriptive message
- May include additional context (e.g., `model_id`, `model_name`)

### Common Error Types

#### Model Not Found
```python
{
    'status': 'error',
    'error': 'Model my_model not found',
    'model_name': 'my_model'
}
```

#### Invalid Parameters
```python
{
    'status': 'error', 
    'error': 'Unknown model type: invalid_type. Available types: [...]'
}
```

#### Missing Features
```python
{
    'status': 'error',
    'error': 'Missing required features: [\'koi_period\', \'koi_prad\']'
}
```

#### Model Not Loaded
```python
{
    'status': 'error',
    'error': 'Model my_model not loaded',
    'model_id': 'my_model'
}
```

### Best Practices

1. **Always check status/success** before processing results
2. **Handle missing features** by checking dataset requirements
3. **Validate model types** using `list_available_models()`
4. **Use try-catch blocks** for API calls in production
5. **Share PredictionAPI instances** between ExplanationAPI for consistency

### Example Error Handling
```python
try:
    result = api.train_model('random_forest', 'kepler')
    
    if result.get('success'):
        print(f"Model trained: {result['model_name']}")
        
        # Make prediction
        pred_result = api.predict_single(
            result['model_name'], 
            sample_features
        )
        
        if pred_result.get('success'):
            print(f"Prediction: {pred_result['prediction']}")
        else:
            print(f"Prediction failed: {pred_result.get('error')}")
    else:
        print(f"Training failed: {result.get('error')}")
        
except Exception as e:
    print(f"Unexpected error: {e}")
```

This completes the comprehensive API documentation for the NASA Exoplanet ML System. Each API provides detailed parameter specifications, return value structures, and practical examples for effective usage.