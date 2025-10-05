# Exoplanet Playground - Backend

Flask backend API for the Exoplanet Playground web application.

## Overview

This backend provides REST API endpoints for:
- **Custom Model Training**: Train ML models with user-selected configurations
- **Pretrained Model Prediction**: Run predictions using pretrained models
- **Progress Tracking**: Monitor training/prediction progress
- **Results Delivery**: Provide prediction results with explanations

## Architecture

```
backend/
├── app.py                 # Main Flask application with API endpoints
├── config.py              # Configuration settings
├── requirements.txt       # Backend dependencies
└── utils/
    ├── session_manager.py # Session state management
    ├── ml_integration.py  # ML API wrapper
    └── file_handler.py    # File upload handling
```

## API Endpoints

### Custom Training Flow

1. **POST /api/custom/train**
   - Starts model training with user configuration
   - Accepts: model type, dataset selection, hyperparameters, uploaded files
   - Returns: `session_id`

2. **GET /api/custom/progress/:sessionId**
   - Returns training progress (0-100%) and current step
   - Status: 'running' | 'completed' | 'error'

3. **GET /api/custom/result/:sessionId**
   - Returns validation metrics, test predictions, and explanations
   - Includes confusion matrix, accuracy, feature importance

### Pretrained Model Flow

1. **POST /api/pretrained/predict**
   - Runs pretrained model (with optional fine-tuning)
   - Accepts: dataset selection or uploaded data for fine-tuning
   - Returns: `session_id`

2. **GET /api/pretrained/progress/:sessionId**
   - Returns prediction/fine-tuning progress

3. **GET /api/pretrained/result/:sessionId**
   - Returns validation metrics and test predictions

### Utility Endpoints

- **GET /api/datasets** - List available NASA datasets
- **GET /api/models** - List available model types
- **GET /api/health** - Health check

## Integration Points

### Frontend → Backend
- User selections (model, dataset, hyperparameters)
- File uploads (training/testing data)
- Progress polling
- Result retrieval

### Backend → ML APIs
- `TrainingAPI`: Manage training sessions and data preparation
- `PredictionAPI`: Load models and make predictions
- `ExplanationAPI`: Generate feature importance and explanations

## Data Flow

### Custom Training
1. Frontend submits training configuration
2. Backend saves uploaded files (if any)
3. Backend creates training session via `TrainingAPI`
4. Backend loads and prepares data
5. Backend starts training asynchronously
6. Frontend polls progress endpoint
7. Backend returns validation metrics + test predictions

### Pretrained Prediction
1. Frontend submits dataset choice or uploaded data
2. Backend loads pretrained model
3. If user data: fine-tune model with uploaded data
4. Backend runs predictions on test data
5. Backend generates explanations
6. Frontend retrieves results

## Key Features

- **Session Management**: Track multiple concurrent training/prediction sessions
- **File Handling**: Secure file upload and validation
- **Progress Tracking**: Real-time training progress updates
- **ML Integration**: Seamless integration with existing ML APIs
- **Explanation Generation**: Feature importance and prediction explanations

## Implementation Status

🟡 **Prototype Phase** - Core structure implemented, details to be completed

### Completed
- API endpoint structure
- Session manager skeleton
- ML integration wrapper skeleton
- File handler skeleton
- Configuration setup

### To Be Implemented
- Actual ML API integration logic
- Session state persistence
- Progress tracking implementation
- Result formatting and delivery
- Error handling and validation
- File validation logic
- Session cleanup

## Running the Backend

```bash
# Install dependencies (from project root)
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Run the server
cd backend
python app.py
```

Server will run on `http://localhost:5000`

## Environment Variables

(To be configured as needed)

```
FLASK_ENV=development
FLASK_DEBUG=1
```

## Next Steps

1. **Implement ML Integration**: Complete `ml_integration.py` to call TrainingAPI and PredictionAPI
2. **Implement Session Management**: Complete session state tracking and persistence
3. **Implement File Handling**: Complete file validation and processing
4. **Add Error Handling**: Comprehensive error handling and validation
5. **Testing**: Add unit tests and integration tests
6. **Frontend Integration**: Connect frontend forms to backend endpoints
7. **Async Processing**: Consider background job processing for long-running tasks

## Notes

- Backend designed for minimal code and maximum readability
- Uses existing ML APIs without reimplementing logic
- Follows RESTful API design patterns
- CORS enabled for frontend communication
- File upload size limit: 100MB
