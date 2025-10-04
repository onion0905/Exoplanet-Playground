# Exoplanet Playground Backend

This is the Flask backend for the Exoplanet Machine Learning Web Application. It provides REST API endpoints for training machine learning models to identify exoplanets using NASA data.

## Features

- 🚀 **Model Training**: Train multiple ML algorithms (Random Forest, XGBoost, Neural Networks, etc.)
- 📊 **NASA Datasets**: Support for Kepler, K2, and TESS datasets
- 📁 **Custom Data**: Upload your own CSV datasets
- ⚙️ **Hyperparameter Tuning**: Configure model parameters through the API
- 📈 **Progress Tracking**: Real-time training progress updates
- 🔍 **Model Explanation**: Feature importance and prediction explanations
- 🎯 **Predictions**: Make predictions on new data with confidence scores

## API Endpoints

### Core Pages
- `GET /` - Homepage with project information
- `GET/POST /select` - Model and data selection interface
- `GET/POST /training` - Training process and progress tracking
- `GET/POST /predict` - Prediction data upload interface
- `GET/POST /result` - Results visualization and explanations

### Additional APIs
- `GET /api/datasets` - Available NASA datasets information
- `GET /api/models` - Available ML models information
- `POST /api/upload` - File upload for custom data
- `GET /api/sessions/<id>` - Session management
- `GET /api/health` - Health check endpoint

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Backend**
   ```bash
   python run.py
   ```
   
   Or directly:
   ```bash
   python backend/app.py
   ```

3. **Access the API**
   - Backend runs on `http://localhost:5000`
   - Visit homepage: `http://localhost:5000/`
   - API endpoints: `http://localhost:5000/api/...`

## Project Structure

```
backend/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
uploads/                # User uploaded files
ML/                     # Machine Learning modules
├── src/
│   ├── api/           # ML API interfaces
│   │   ├── training_api.py
│   │   ├── prediction_api.py
│   │   └── explanation_api.py
│   ├── models/        # ML model implementations
│   ├── data/          # Data processing utilities
│   └── explainability/ # Model explanation tools
```

## Configuration

The backend uses environment-based configuration:

- `FLASK_ENV`: Set to 'development' or 'production'
- `PORT`: Server port (default: 5000)
- `SECRET_KEY`: Flask secret key for sessions

## Integration with Frontend

This backend is designed to work with the React frontend in the `frontend/` directory. The frontend makes API calls to:

1. **Select Page**: Configure training sessions via `/select`
2. **Training Page**: Monitor progress via `/training`
3. **Predict Page**: Upload test data via `/predict`
4. **Result Page**: View results and explanations via `/result`

## Development

### Session Management

The backend uses session-based training where each user workflow gets a unique session ID to track:
- Data selection (NASA datasets or uploaded files)
- Model configuration and hyperparameters
- Training progress and results
- Prediction data and results

### ML Integration

The backend integrates with the ML module through three main APIs:
- **TrainingAPI**: Handles model training workflows
- **PredictionAPI**: Makes predictions with trained models
- **ExplanationAPI**: Generates model explanations

### Error Handling

All endpoints return standardized JSON responses:
```json
{
  "status": "success|error",
  "data": {...},
  "error": "error message if applicable"
}
```

## Production Deployment

For production deployment:

1. Set `FLASK_ENV=production`
2. Configure proper `SECRET_KEY`
3. Use a production WSGI server (gunicorn, uWSGI)
4. Set up proper logging and monitoring
5. Configure CORS for your frontend domain

Example with gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## API Examples

### Start a Training Session
```bash
# Get new session
curl http://localhost:5000/select

# Configure data source
curl -X POST http://localhost:5000/select \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "config_type": "data_source",
    "data_source": "nasa",
    "dataset": "kepler"
  }'

# Start training
curl -X POST http://localhost:5000/training \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id"}'
```

### Check Training Progress
```bash
curl "http://localhost:5000/training?session_id=your-session-id"
```

### Get Results
```bash
curl "http://localhost:5000/result?session_id=your-session-id"
```