# API Usage Examples for Exoplanet ML Backend

This document provides examples of how to use the enhanced backend API for training and prediction.

## 1. Training with NASA Dataset

### Step 1: Validate Configuration
```bash
POST /api/validate-config
Content-Type: application/json

{
    "model_type": "random_forest",
    "dataset_source": "nasa",
    "dataset_name": "kepler",
    "target_column": "koi_disposition",
    "test_size": 0.2,
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
    }
}
```

### Step 2: Start Training
```bash
POST /api/training/start
Content-Type: application/json

{
    "model_type": "random_forest",
    "dataset_source": "nasa",
    "dataset_name": "kepler",
    "target_column": "koi_disposition",
    "test_size": 0.2,
    "model_name": "kepler_rf_model",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    }
}
```

### Step 3: Monitor Progress
```bash
GET /api/training/progress/{session_id}
```

Response:
```json
{
    "session_id": "session_abc123",
    "status": "completed",
    "progress": 100,
    "current_step": "Training completed successfully!",
    "ready_for_results": true,
    "results_summary": {
        "model_name": "kepler_rf_model",
        "accuracy": 0.85,
        "total_samples": 1000,
        "feature_count": 25
    }
}
```

### Step 4: Get Complete Results
```bash
GET /api/training/results/{session_id}
```

Response includes:
- Training metrics
- Testing results with confusion matrix
- Feature importance
- Prediction accuracy
- Model summary

## 2. Training with Custom Dataset (Single File)

### Step 1: Upload Dataset
```bash
POST /api/upload
Content-Type: multipart/form-data

upload_type: single_file
data_file: [your_dataset.csv]
```

### Step 2: Start Training
```bash
POST /api/training/start
Content-Type: application/json

{
    "model_type": "xgboost",
    "dataset_source": "upload",
    "uploaded_files": {
        "data_file": "uploaded_filename_from_step1.csv"
    },
    "target_column": "is_exoplanet",
    "test_size": 0.25,
    "model_name": "custom_xgboost_model"
}
```

## 3. Training with Separate Train/Test Files

### Step 1: Upload Files
```bash
POST /api/upload
Content-Type: multipart/form-data

upload_type: separate_files
training_file: [training_data.csv]
testing_file: [testing_data.csv]
```

### Step 2: Start Training
```bash
POST /api/training/start
Content-Type: application/json

{
    "model_type": "deep_learning",
    "dataset_source": "upload",
    "upload_type": "separate_files",
    "uploaded_files": {
        "training_file": "train_filename.csv",
        "testing_file": "test_filename.csv"
    },
    "target_column": "classification",
    "model_name": "custom_nn_model",
    "hyperparameters": {
        "hidden_layers": [64, 32],
        "learning_rate": 0.001,
        "epochs": 50
    }
}
```

## 4. Making Predictions

### Single Prediction
```bash
POST /api/training/predict/{session_id}
Content-Type: application/json

{
    "features": {
        "koi_period": 10.5,
        "koi_impact": 0.2,
        "koi_duration": 3.5,
        "koi_depth": 100,
        "koi_prad": 1.2
    }
}
```

Response:
```json
{
    "session_id": "session_abc123",
    "prediction_result": {
        "prediction": "CONFIRMED",
        "confidence": 0.87,
        "probabilities": {
            "CONFIRMED": 0.87,
            "FALSE POSITIVE": 0.13
        }
    }
}
```

## 5. Available Model Types and Hyperparameters

### Random Forest
```json
{
    "model_type": "random_forest",
    "hyperparameters": {
        "n_estimators": 100,          // Number of trees
        "max_depth": 10,              // Maximum depth of trees
        "min_samples_split": 2,       // Minimum samples to split
        "min_samples_leaf": 1         // Minimum samples in leaf
    }
}
```

### XGBoost
```json
{
    "model_type": "xgboost", 
    "hyperparameters": {
        "n_estimators": 100,          // Number of boosting rounds
        "learning_rate": 0.1,         // Step size shrinkage
        "max_depth": 6,               // Maximum tree depth
        "subsample": 0.8              // Subsample ratio
    }
}
```

### Support Vector Machine
```json
{
    "model_type": "svm",
    "hyperparameters": {
        "C": 1.0,                     // Regularization parameter
        "kernel": "rbf",              // Kernel type
        "gamma": "scale"              // Kernel coefficient
    }
}
```

### Deep Learning
```json
{
    "model_type": "deep_learning",
    "hyperparameters": {
        "hidden_layers": [64, 32, 16], // Hidden layer sizes
        "learning_rate": 0.001,         // Learning rate
        "batch_size": 32,               // Batch size
        "epochs": 100                   // Training epochs
    }
}
```

## 6. Expected Results Format

When training completes, the results include:

### Summary Metrics
```json
{
    "summary_metrics": {
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.87,
        "f1_score": 0.85,
        "total_samples": 1000,
        "correct_predictions": 850,
        "incorrect_predictions": 150
    }
}
```

### Confusion Matrix
```json
{
    "confusion_matrix": {
        "matrix": [[450, 50], [100, 400]],
        "labels": ["CONFIRMED", "FALSE POSITIVE"],
        "matrix_with_labels": [
            {
                "true_label": "CONFIRMED",
                "predictions": {
                    "CONFIRMED": 450,
                    "FALSE POSITIVE": 50
                }
            }
        ]
    }
}
```

### Feature Importance
```json
{
    "feature_importance": {
        "available": true,
        "type": "gini_importance",
        "top_5_features": [
            {"feature": "koi_period", "importance": 0.25},
            {"feature": "koi_duration", "importance": 0.20},
            {"feature": "koi_depth", "importance": 0.18}
        ]
    }
}
```

## 7. Frontend Integration

The backend provides signals for frontend navigation:

- `ready_for_results: true` indicates training is complete
- Frontend should redirect to `/custom_result` when this flag is set
- Use `/api/training/results/{session_id}` to get complete results for the results page
- The results include all data needed for visualization and analysis

## 8. Error Handling

All endpoints return consistent error responses:

```json
{
    "error": "Descriptive error message",
    "details": "Additional error context if available"
}
```

Common HTTP status codes:
- 200: Success
- 400: Bad request (invalid configuration, missing data)
- 404: Resource not found (session, file)
- 500: Internal server error