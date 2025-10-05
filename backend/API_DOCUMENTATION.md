# Backend API Documentation

## Base URL
```
http://localhost:5000/api
```

## Endpoints

### 1. Custom Model Training

#### POST /api/custom/train
Start custom model training with user-selected configuration.

**Request Format:**
- Content-Type: `multipart/form-data` (when uploading files) or `application/json`

**Request Body:**
```json
{
  "model_type": "random_forest",           // Required: Model type
  "data_source": "nasa",                    // Required: 'nasa' or 'user'
  
  // If data_source = 'nasa':
  "dataset_name": "kepler",                 // Required: 'kepler', 'tess', 'k2'
  
  // If data_source = 'user':
  "data_format": "kepler",                  // Required: Format of uploaded data
  "training_file": <File>,                  // Required: Training CSV file
  "testing_file": <File>,                   // Required: Testing CSV file
  
  // Hyperparameters (optional, model-specific):
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2
  }
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc123",
  "message": "Training started"
}
```

---

#### GET /api/custom/progress/:sessionId
Get training progress for a session.

**Response:**
```json
{
  "success": true,
  "progress": 75,                           // 0-100
  "current_step": "Training model...",      // Current step description
  "status": "running"                       // 'running' | 'completed' | 'error'
}
```

---

#### GET /api/custom/result/:sessionId
Get training results including validation metrics and test predictions.

**Response:**
```json
{
  "success": true,
  "metrics": {
    "accuracy": 0.92,
    "f1_score": 0.89,
    "precision": 0.91,
    "recall": 0.88,
    "confusion_matrix": [[50, 5], [3, 42]]  // From validation set
  },
  "predictions": [
    {
      "id": 1,
      "name": "Kepler-452b",
      "prediction": "Exoplanet",
      "confidence": 0.94,
      "probabilities": {
        "Exoplanet": 0.94,
        "False Positive": 0.06
      },
      "feature_importance": {
        "transit_depth": 0.25,
        "orbital_period": 0.18,
        "stellar_radius": 0.15
      },
      "reasons": [
        "Strong transit signal detected",
        "Orbital period indicates stable orbit"
      ]
    }
  ],
  "model_info": {
    "model_type": "random_forest",
    "dataset_used": "kepler",
    "training_time": 45.2,
    "feature_count": 15
  }
}
```

---

### 2. Pretrained Model Prediction

#### POST /api/pretrained/predict
Run pretrained model prediction (with optional fine-tuning).

**Request Format:**
- Content-Type: `multipart/form-data` (when uploading files) or `application/json`

**Request Body:**
```json
{
  "data_source": "nasa",                    // Required: 'nasa' or 'user'
  
  // If data_source = 'nasa':
  "dataset_name": "kepler",                 // Required: 'kepler', 'tess', 'k2'
  
  // If data_source = 'user' (fine-tuning):
  "data_format": "kepler",                  // Required: Format of uploaded data
  "training_file": <File>                   // Required: Data to fine-tune with
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "xyz789",
  "message": "Prediction started"
}
```

---

#### GET /api/pretrained/progress/:sessionId
Get prediction/fine-tuning progress.

**Response:** (Same format as custom/progress)

---

#### GET /api/pretrained/result/:sessionId
Get pretrained model results.

**Response:** (Same format as custom/result)

---

### 3. Utility Endpoints

#### GET /api/datasets
List available NASA datasets.

**Response:**
```json
{
  "success": true,
  "datasets": ["kepler", "tess", "k2"]
}
```

---

#### GET /api/models
List available model types.

**Response:**
```json
{
  "success": true,
  "models": [
    "random_forest",
    "decision_tree",
    "xgboost",
    "svm",
    "linear_regression",
    "deep_learning",
    "pca"
  ]
}
```

---

#### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "message": "Exoplanet Playground API is running"
}
```

---

## Frontend Integration Guide

### Custom Training Flow

```javascript
// Step 1: Start Training
const formData = new FormData();
formData.append('model_type', 'random_forest');
formData.append('data_source', 'user');
formData.append('data_format', 'kepler');
formData.append('training_file', trainingFile);
formData.append('testing_file', testingFile);
formData.append('hyperparameters', JSON.stringify({
  n_estimators: 100,
  max_depth: 10
}));

const response = await fetch('http://localhost:5000/api/custom/train', {
  method: 'POST',
  body: formData
});
const { session_id } = await response.json();

// Step 2: Navigate to Progress Page
navigate(`/custom/progress?session=${session_id}`);

// Step 3: Poll Progress
const pollProgress = setInterval(async () => {
  const res = await fetch(`http://localhost:5000/api/custom/progress/${session_id}`);
  const data = await res.json();
  
  setProgress(data.progress);
  setCurrentStep(data.current_step);
  
  if (data.status === 'completed') {
    clearInterval(pollProgress);
    navigate(`/custom/result?session=${session_id}`);
  }
}, 1000);

// Step 4: Get Results
const resultRes = await fetch(`http://localhost:5000/api/custom/result/${session_id}`);
const results = await resultRes.json();

// Display metrics and predictions
setMetrics(results.metrics);
setPredictions(results.predictions);
```

### Pretrained Prediction Flow

```javascript
// Similar flow but simpler (no hyperparameters)
const formData = new FormData();
formData.append('data_source', 'nasa');
formData.append('dataset_name', 'kepler');

const response = await fetch('http://localhost:5000/api/pretrained/predict', {
  method: 'POST',
  body: formData
});

// Then follow same progress polling and result retrieval
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "success": false,
  "error": "Error message description"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `202` - Accepted (training/prediction started)
- `400` - Bad Request (invalid input)
- `404` - Not Found (session/endpoint not found)
- `500` - Internal Server Error

---

## Model-Specific Hyperparameters

### Random Forest
```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 1
}
```

### XGBoost
```json
{
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "subsample": 0.8
}
```

### SVM
```json
{
  "C": 1.0,
  "kernel": "rbf",
  "gamma": "scale"
}
```

### Decision Tree
```json
{
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 1
}
```

### Deep Learning
```json
{
  "hidden_layers": [64, 32],
  "learning_rate": 0.001,
  "dropout_rate": 0.2,
  "epochs": 100
}
```

---

## Data Format Requirements

### CSV File Format
Files must be CSV format with:
- Header row with column names
- Numeric values for features
- Target column matching dataset type:
  - Kepler: `koi_disposition`
  - TESS: `tfopwg_disp`
  - K2: `disposition`

### Accepted Target Values
- `CONFIRMED` / `Confirmed` → Exoplanet
- `FALSE POSITIVE` / `False Positive` → Not an exoplanet
- `CANDIDATE` / `Candidate` → Needs verification

---

## Implementation Notes for Developers

### Backend Implementation Tasks

1. **ml_integration.py**
   - Integrate with `TrainingAPI` for training sessions
   - Integrate with `PredictionAPI` for predictions
   - Integrate with `ExplanationAPI` for feature importance
   - Format results for frontend consumption

2. **session_manager.py**
   - Implement progress tracking
   - Store session results
   - Handle session timeouts
   - Thread-safe session operations

3. **file_handler.py**
   - Validate CSV format
   - Check data compatibility
   - Clean up temporary files

4. **app.py**
   - Parse multipart/form-data requests
   - Handle file uploads
   - Implement async processing for long tasks
   - Add comprehensive error handling

### Frontend Integration Tasks

1. **Update CustomPage.jsx**
   - Send form data to `/api/custom/train`
   - Navigate to progress page with session_id

2. **Update CustomProgressPage.jsx**
   - Poll `/api/custom/progress/:sessionId`
   - Update progress bar and status
   - Navigate to result page on completion

3. **Update CustomResultPage.jsx**
   - Fetch from `/api/custom/result/:sessionId`
   - Display metrics, predictions, explanations
   - Render confusion matrix visualization

4. **Update PretrainedPage.jsx, PretrainedProgressPage.jsx, PretrainedResultPage.jsx**
   - Similar updates for pretrained flow

---

## Testing Endpoints

Use curl or Postman to test endpoints:

```bash
# Health check
curl http://localhost:5000/api/health

# List datasets
curl http://localhost:5000/api/datasets

# Start training (with NASA data)
curl -X POST http://localhost:5000/api/custom/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "data_source": "nasa",
    "dataset_name": "kepler"
  }'

# Check progress
curl http://localhost:5000/api/custom/progress/abc123

# Get results
curl http://localhost:5000/api/custom/result/abc123
```
