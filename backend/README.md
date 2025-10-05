# NASA Exoplanet ML Backend

A Flask-based REST API backend for the NASA Exoplanet Machine Learning Playground web application.

## Features

- **Dataset Management**: Support for NASA Kepler, TESS, and K2 datasets
- **Model Training**: Multiple ML algorithms (Random Forest, SVM, XGBoost, etc.)
- **Real-time Progress**: WebSocket-like progress tracking for training sessions
- **Custom Data Upload**: Support for user-uploaded CSV/Excel files
- **Predictions & Explanations**: Model predictions with confidence scores and explainability
- **Interactive Results**: Feature importance analysis and visualization data

## API Endpoints

### Homepage & Info
- `GET /api/` - Project information
- `GET /api/status` - System health status
- `GET /api/about` - About information

### Dataset & Model Selection
- `GET /api/datasets` - List available NASA datasets
- `GET /api/models` - List available ML models
- `GET /api/trained-models` - List trained models
- `POST /api/upload` - Upload custom dataset
- `POST /api/validate-config` - Validate training configuration

### Training
- `POST /api/training/start` - Start model training
- `GET /api/training/progress/<session_id>` - Get training progress
- `GET /api/training/status/<session_id>` - Get training status
- `POST /api/training/cancel/<session_id>` - Cancel training
- `GET /api/training/sessions` - List all sessions
- `POST /api/training/cleanup` - Cleanup old sessions

### Results & Predictions
- `GET /api/predictions/<session_id>` - Get prediction results
- `GET /api/explain/<session_id>` - Get model explanations
- `POST /api/explain-prediction` - Explain specific prediction
- `POST /api/predict` - Make custom predictions
- `GET /api/download/<session_id>` - Download results
- `GET /api/visualization/<session_id>` - Get visualization data

## Setup

1. **Install Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Run Development Server**
   ```bash
   python app.py
   ```
   
   The server will start on `http://localhost:5000`

4. **Production Deployment**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## Configuration

The backend supports environment-based configuration:

- **Development**: Debug enabled, verbose logging
- **Production**: Optimized for performance and security
- **Testing**: Isolated environment for automated tests

Key environment variables:
- `SECRET_KEY`: Flask secret key for sessions
- `FLASK_ENV`: Environment (development/production/testing)
- `CORS_ORIGINS`: Allowed frontend origins
- `MAX_UPLOAD_SIZE`: Maximum file upload size
- `SESSION_TIMEOUT`: Training session timeout

## Architecture

```
backend/
├── app.py              # Main Flask application
├── config/             # Configuration management
├── routes/             # API route blueprints
│   ├── home.py        # Homepage endpoints
│   ├── select.py      # Selection page endpoints
│   ├── training.py    # Training endpoints
│   └── results.py     # Results endpoints
├── services/           # Business logic layer
│   ├── ml_service.py  # ML operations wrapper
│   └── session_service.py # Session management
└── middleware/         # Request/response middleware
```

## Integration with ML Module

The backend integrates with the existing ML module through service classes:

- **TrainingAPI**: Handles model training workflows
- **PredictionAPI**: Manages model loading and predictions
- **ExplanationAPI**: Provides model explainability features
- **ExoplanetMLAPI**: User-friendly wrapper for common operations

## Error Handling

The backend includes comprehensive error handling:

- **HTTP Status Codes**: Proper REST API status codes
- **Error Messages**: User-friendly error descriptions
- **Logging**: Detailed server-side logging for debugging
- **Validation**: Input validation and sanitization

## Security Features

- **CORS**: Configurable cross-origin resource sharing
- **File Upload**: Secure file handling with size limits
- **Input Validation**: Request data validation and sanitization
- **Error Masking**: Production error message masking

## Monitoring & Logging

- **Health Checks**: `/health` endpoint for service monitoring
- **Request Logging**: All API requests are logged
- **Performance Tracking**: Request duration tracking
- **Session Management**: Automatic cleanup of old sessions

## Development

For development, the backend supports:

- **Hot Reload**: Automatic restart on code changes
- **Debug Mode**: Detailed error traces and debugging
- **CORS**: Pre-configured for common frontend dev ports
- **Logging**: Verbose logging for development debugging

The backend is designed to work seamlessly with the React frontend and provide a robust foundation for the exoplanet ML playground application.