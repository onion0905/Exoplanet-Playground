# Frontend-Backend Integration Guide

This document describes the complete integration between the frontend React application and the Flask backend API for the Exoplanet Playground project.

## Architecture Overview

```
┌─────────────────┐    HTTP/API    ┌─────────────────┐    Python    ┌─────────────────┐
│   React         │◄──────────────►│   Flask         │◄─────────────►│   ML Module     │
│   Frontend      │   (port 3000)  │   Backend       │              │   (Training/    │
│                 │                │   (port 5000)   │              │    Prediction)  │
└─────────────────┘                └─────────────────┘              └─────────────────┘
```

## Integration Points

### 1. SelectPage → Backend API
**File**: `frontend/src/pages/select/SelectPage.jsx`

**Changes Made**:
- Modified `handleStartTraining()` function to send configuration to backend
- Added file upload handling for custom datasets
- Integrated with `/api/training/start` endpoint
- Added proper error handling and user feedback

**API Call**:
```javascript
// Training configuration sent to backend
const trainingConfig = {
  model_type: 'rf|xgb|nn',
  dataset_source: 'nasa|user',
  dataset_name: 'kepler|tess|k2',  // for NASA datasets
  target_column: 'koi_disposition|tfopwg_disp|disposition',
  hyperparameters: { n_estimators: 100, max_depth: 10, ... },
  uploaded_files: { ... },  // for user uploads
  upload_type: 'separate_files|single_file'
};
```

### 2. TrainingPage → Real-time Progress Tracking
**File**: `frontend/src/pages/training/TrainingPage.jsx`

**Changes Made**:
- Replaced simulation with real backend polling
- Integrated with `/api/training/progress` endpoint
- Added session ID management via localStorage
- Real-time display of training configuration and progress

**Progress Polling**:
```javascript
// Poll backend every 2 seconds for progress updates
const pollProgress = async () => {
  const response = await fetch(`/api/training/progress?session_id=${sessionId}`);
  const data = await response.json();
  
  setProgress(data.progress);
  setCurrentStep(data.current_step);
  
  if (data.completed) {
    localStorage.setItem('trainingResults', JSON.stringify(data));
    navigate("/custom_result");
  }
};
```

### 3. CustomResultPage → Backend Results Display
**File**: `frontend/src/pages/result/CustomResultPage.jsx`

**Changes Made**:
- Added real results loading from backend
- Integrated confusion matrix display
- Added model performance metrics
- Dynamic results table with explainable ML features

**Results Structure**:
```javascript
// Results from backend
{
  metrics: {
    accuracy: 0.85,
    precision: 0.82,
    recall: 0.88,
    f1_score: 0.85
  },
  confusion_matrix: {
    true_positive: 120,
    false_positive: 15,
    true_negative: 200,
    false_negative: 25
  },
  test_data: [
    {
      identifier: "Object_1",
      prediction: 1,
      confidence: 0.94,
      actual_label: 1,
      explanation: ["Feature analysis reasons..."]
    }
  ]
}
```

## Backend API Endpoints

### Core Training Workflow

1. **POST /api/training/start**
   - Accepts training configuration
   - Starts background training process
   - Returns session ID for tracking

2. **GET /api/training/progress**
   - Query param: `session_id`
   - Returns current progress and status
   - Includes completion flag

3. **GET /api/training/results**
   - Query param: `session_id`
   - Returns comprehensive results with metrics
   - Includes test predictions and explanations

4. **POST /api/upload**
   - Handles file uploads (CSV/Excel)
   - Supports single file or separate train/test files
   - Returns file validation results

### Data Flow

```
User Selection → API Call → Background Training → Progress Updates → Results Display
      ↓              ↓             ↓                    ↓               ↓
  SelectPage → /training/start → ML Module → /training/progress → ResultsPage
      ↓              ↓             ↓                    ↓               ↓
  localStorage ← Session ID ← Training Job ← Real Progress ← Final Results
```

## Configuration

### Frontend Proxy (vite.config.js)
```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
```

### Backend CORS (app.py)
```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
```

## Session Management

### Frontend Session Tracking
- Session ID stored in `localStorage` after training start
- Used for progress polling and results retrieval
- Automatically cleaned up after completion

### Backend Session Management
- In-memory session store for active training jobs
- Thread-safe progress updates
- Automatic cleanup of completed sessions

## Error Handling

### Frontend Error States
- Network connectivity issues
- Invalid configuration validation
- Backend service unavailability
- File upload errors

### Backend Error Responses
```json
{
  "error": "Error description",
  "details": "Additional context",
  "status": "error"
}
```

## Development Workflow

### 1. Setup Development Environment
```bash
# Run the setup script
python setup_dev_environment.py

# Or manual setup:
# Backend
cd backend
pip install -r requirements.txt
python app.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### 2. Test Integration
```bash
# Run comprehensive integration test
python scripts/integration/test_frontend_backend.py
```

### 3. Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- Health Check: http://localhost:5000/health

## Key Features

### Real-time Training Progress
- Live progress updates during model training
- Step-by-step training process visibility
- Estimated completion times

### Comprehensive Results Display
- Model performance metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Individual prediction results with confidence scores
- Explainable ML integration (when available)

### File Upload Support
- Custom dataset uploads (CSV/Excel)
- Single file or separate train/test files
- Real-time validation and feedback

### Session Persistence
- Training state maintained across page refreshes
- Automatic redirect to appropriate pages
- Progress preservation during navigation

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure backend CORS is configured for frontend URL
   - Check proxy configuration in vite.config.js

2. **API Connectivity**
   - Verify both servers are running on correct ports
   - Check firewall settings
   - Confirm proxy configuration

3. **File Upload Issues**
   - Check file format (CSV/Excel)
   - Verify file size limits
   - Ensure proper column headers

4. **Training Failures**
   - Check dataset compatibility
   - Verify ML module installation
   - Monitor backend logs for errors

### Debug Tools

1. **Browser Developer Tools**
   - Network tab for API request inspection
   - Console for JavaScript errors
   - Application tab for localStorage inspection

2. **Backend Logs**
   - Flask debug output
   - ML module training logs
   - Error stack traces

3. **Integration Test**
   - Run `test_frontend_backend.py` for full system test
   - Individual endpoint testing
   - Performance monitoring

## Production Considerations

### Security
- Add authentication/authorization
- Validate all user inputs
- Secure file upload handling
- Rate limiting for API endpoints

### Performance
- Database for session management
- Caching for model results
- Async processing for large datasets
- Load balancing for multiple users

### Monitoring
- API response time monitoring
- Training job success/failure tracking
- User interaction analytics
- Error rate monitoring

## Future Enhancements

1. **Real-time Collaboration**
   - WebSocket integration for live updates
   - Shared training sessions
   - Multi-user result comparison

2. **Enhanced ML Integration**
   - Model versioning and comparison
   - Hyperparameter optimization
   - Cross-validation results

3. **Advanced Visualization**
   - Interactive 3D planet models
   - Feature importance plots
   - Training convergence graphs

4. **Data Management**
   - Dataset versioning
   - Result history and comparison
   - Export/import functionality