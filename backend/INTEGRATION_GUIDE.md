# Backend Integration Guide

This guide explains how to integrate the backend with the frontend and ML components.

## Architecture Overview

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│             │         │             │         │             │
│  Frontend   │────────▶│   Backend   │────────▶│   ML APIs   │
│  (React)    │◀────────│   (Flask)   │◀────────│  (Python)   │
│             │         │             │         │             │
└─────────────┘         └─────────────┘         └─────────────┘
      │                       │                        │
      │                       │                        │
      ▼                       ▼                        ▼
  User Input            Session Mgmt            Model Training
  Display Results       File Handling          Predictions
                        API Routing            Explanations
```

## Integration Tasks

### Phase 1: Backend ML Integration

**File: `backend/utils/ml_integration.py`**

#### Task 1.1: Implement `start_custom_training()`

```python
def start_custom_training(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Start custom model training."""
    
    # 1. Start training session
    self.training_api.start_training_session(session_id)
    
    # 2. Load data
    if config['data_source'] == 'nasa':
        data_config = {'datasets': [config['dataset_name']]}
        self.training_api.load_data_for_training(
            session_id, 
            data_source='nasa',
            data_config=data_config
        )
    else:  # user uploaded data
        data_config = {'filepath': config['train_data_path']}
        self.training_api.load_data_for_training(
            session_id,
            data_source='user',
            data_config=data_config
        )
    
    # 3. Configure training
    training_config = {
        'model_type': config['model_type'],
        'target_column': self._get_target_column(config),
        'hyperparameters': config.get('hyperparameters', {})
    }
    self.training_api.configure_training(session_id, training_config)
    
    # 4. Start training (async recommended)
    result = self.training_api.start_training(session_id)
    
    return result
```

#### Task 1.2: Implement `get_custom_training_result()`

```python
def get_custom_training_result(self, session_id: str) -> Optional[Dict[str, Any]]:
    """Get training results with predictions."""
    
    # 1. Get session info
    session_info = self.training_api.get_session_info(session_id)
    
    if session_info['status'] != 'completed':
        return None
    
    session = session_info['session_info']
    
    # 2. Get validation metrics (from training)
    validation_metrics = {
        'accuracy': session.get('validation_accuracy'),
        'confusion_matrix': session.get('training_metrics', {}).get('confusion_matrix'),
        # Add more metrics
    }
    
    # 3. Get test predictions
    model = session['model']
    X_test = session['prepared_data']['X_test']
    y_test = session['prepared_data']['y_test']
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # 4. Generate explanations for each prediction
    prediction_results = []
    for i in range(len(predictions)):
        instance_data = X_test.iloc[i].to_dict()
        
        # Get explanation
        explanation = self.explanation_api.explain_prediction_local(
            model_id=session_id,
            instance_data=instance_data
        )
        
        prediction_results.append({
            'id': i,
            'prediction': predictions[i],
            'confidence': probabilities[i].max(),
            'probabilities': dict(zip(model.target_classes, probabilities[i])),
            'feature_importance': explanation.get('feature_contributions', {}),
            'actual': y_test.iloc[i]
        })
    
    return {
        'metrics': validation_metrics,
        'predictions': prediction_results,
        'model_info': model.get_model_info()
    }
```

#### Task 1.3: Implement Pretrained Methods

```python
def run_pretrained_prediction(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run pretrained model with optional fine-tuning."""
    
    # 1. Load pretrained model
    pretrained_model_path = str(PRETRAINED_MODEL_PATH / 'pretrained_model.joblib')
    self.prediction_api.load_model(pretrained_model_path, model_id='pretrained')
    
    # 2. If user uploaded data for fine-tuning
    if config['data_source'] == 'user':
        # Fine-tune the model
        # Use training_api to retrain with user data + NASA data
        pass
    
    # 3. Run predictions on test data
    # Similar to custom training result generation
    
    return result
```

### Phase 2: Backend Session Management

**File: `backend/utils/session_manager.py`**

#### Task 2.1: Implement Progress Updates

```python
def update_progress(self, session_id: str, progress: int, current_step: str, status: str = 'running'):
    """Update session progress."""
    with self._lock:
        if session_id in self.sessions:
            self.sessions[session_id].update({
                'progress': progress,
                'current_step': current_step,
                'status': status,
                'updated_at': datetime.now().isoformat()
            })
```

#### Task 2.2: Implement Result Storage

```python
def set_result(self, session_id: str, result: Dict[str, Any]):
    """Store session result."""
    with self._lock:
        if session_id in self.sessions:
            self.sessions[session_id]['result'] = result
            self.sessions[session_id]['status'] = 'completed'
            self.sessions[session_id]['progress'] = 100
            self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
```

### Phase 3: Backend Request Handling

**File: `backend/app.py`**

#### Task 3.1: Implement File Upload Handling

```python
@app.route('/api/custom/train', methods=['POST'])
def custom_train():
    """Start custom model training."""
    try:
        session_id = str(uuid.uuid4())
        
        # Parse form data
        model_type = request.form.get('model_type')
        data_source = request.form.get('data_source')
        
        config = {
            'model_type': model_type,
            'data_source': data_source
        }
        
        if data_source == 'nasa':
            config['dataset_name'] = request.form.get('dataset_name')
        else:
            # Handle file uploads
            training_file = request.files.get('training_file')
            testing_file = request.files.get('testing_file')
            
            train_info = file_handler.save_uploaded_file(training_file, 'train')
            test_info = file_handler.save_uploaded_file(testing_file, 'test')
            
            config['train_data_path'] = train_info['filepath']
            config['test_data_path'] = test_info['filepath']
            config['data_format'] = request.form.get('data_format')
        
        # Parse hyperparameters
        if request.form.get('hyperparameters'):
            config['hyperparameters'] = json.loads(request.form.get('hyperparameters'))
        
        # Create session
        session_manager.create_session(session_id, 'custom_training', config)
        
        # Start training (in background thread)
        import threading
        thread = threading.Thread(
            target=self._run_custom_training,
            args=(session_id, config)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Training started'
        }), 202
        
    except Exception as e:
        logger.error(f"Error in custom_train: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def _run_custom_training(self, session_id: str, config: Dict[str, Any]):
    """Background task for training."""
    try:
        # Update progress
        session_manager.update_progress(session_id, 10, 'Loading data...')
        
        # Start training
        result = ml_integration.start_custom_training(session_id, config)
        
        # Poll training progress from ML API
        while True:
            progress_info = self.training_api.training_progress.get(session_id)
            if progress_info:
                session_manager.update_progress(
                    session_id,
                    progress_info['progress'],
                    progress_info['current_step'],
                    progress_info['status']
                )
                
                if progress_info['status'] == 'completed':
                    break
            time.sleep(1)
        
        # Get results
        final_result = ml_integration.get_custom_training_result(session_id)
        session_manager.set_result(session_id, final_result)
        
    except Exception as e:
        logger.error(f"Error in background training: {str(e)}")
        session_manager.set_error(session_id, str(e))
```

#### Task 3.2: Implement Progress/Result Endpoints

```python
@app.route('/api/custom/progress/<session_id>', methods=['GET'])
def custom_progress(session_id):
    """Get training progress."""
    try:
        progress_info = session_manager.get_progress(session_id)
        
        if not progress_info:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            **progress_info
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/custom/result/<session_id>', methods=['GET'])
def custom_result(session_id):
    """Get training results."""
    try:
        result = session_manager.get_result(session_id)
        
        if not result:
            session = session_manager.get_session(session_id)
            if not session:
                return jsonify({'success': False, 'error': 'Session not found'}), 404
            if session['status'] != 'completed':
                return jsonify({'success': False, 'error': 'Training not completed'}), 400
        
        return jsonify({
            'success': True,
            **result
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

### Phase 4: Frontend Integration

**File: `frontend/src/pages/select/CustomPage.jsx`**

#### Task 4.1: Update handleStartTraining

```javascript
const handleStartTraining = async () => {
  try {
    const formData = new FormData();
    formData.append('model_type', selectedModel);
    formData.append('data_source', dataSource);
    
    if (dataSource === 'nasa') {
      formData.append('dataset_name', selectedDataset);
    } else {
      formData.append('data_format', selectedTrainingFormat);
      formData.append('training_file', uploadedFile);
      formData.append('testing_file', uploadedTestFile);
    }
    
    formData.append('hyperparameters', JSON.stringify(hyperparameters));
    
    const response = await fetch('http://localhost:5000/api/custom/train', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.success) {
      // Navigate to progress page with session_id
      navigate(`/custom/progress?session=${data.session_id}`);
    } else {
      alert('Error: ' + data.error);
    }
  } catch (error) {
    console.error('Error starting training:', error);
    alert('Failed to start training');
  }
};
```

**File: `frontend/src/pages/progress/CustomProgressPage.jsx`**

#### Task 4.2: Update Progress Polling

```javascript
useEffect(() => {
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get('session');
  
  if (!sessionId) {
    navigate('/custom');
    return;
  }
  
  const pollProgress = setInterval(async () => {
    try {
      const response = await fetch(`http://localhost:5000/api/custom/progress/${sessionId}`);
      const data = await response.json();
      
      if (data.success) {
        setProgress(data.progress);
        setCurrentStep(data.current_step);
        
        if (data.status === 'completed') {
          clearInterval(pollProgress);
          setIsComplete(true);
          setTimeout(() => {
            navigate(`/custom/result?session=${sessionId}`);
          }, 2000);
        } else if (data.status === 'error') {
          clearInterval(pollProgress);
          alert('Training failed');
          navigate('/custom');
        }
      }
    } catch (error) {
      console.error('Error polling progress:', error);
    }
  }, 1000);
  
  return () => clearInterval(pollProgress);
}, [navigate]);
```

**File: `frontend/src/pages/result/CustomResultPage.jsx`**

#### Task 4.3: Fetch and Display Results

```javascript
useEffect(() => {
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get('session');
  
  if (!sessionId) {
    navigate('/custom');
    return;
  }
  
  const fetchResults = async () => {
    try {
      const response = await fetch(`http://localhost:5000/api/custom/result/${sessionId}`);
      const data = await response.json();
      
      if (data.success) {
        setMetrics(data.metrics);
        setPredictions(data.predictions);
        setModelInfo(data.model_info);
      } else {
        alert('Error fetching results: ' + data.error);
        navigate('/custom');
      }
    } catch (error) {
      console.error('Error fetching results:', error);
      alert('Failed to load results');
      navigate('/custom');
    }
  };
  
  fetchResults();
}, [navigate]);

// Update the predictionResults state with fetched data
const [predictionResults, setPredictionResults] = useState([]);
const [metrics, setMetrics] = useState(null);
const [modelInfo, setModelInfo] = useState(null);

// Format predictions for display
const formatPredictions = (predictions) => {
  return predictions.map((pred, idx) => ({
    id: idx + 1,
    name: `Object-${idx + 1}`,
    prediction: pred.prediction,
    confidence: pred.confidence,
    reasons: Object.entries(pred.feature_importance || {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([feature, importance]) => 
        `${feature}: ${(importance * 100).toFixed(1)}% importance`
      ),
    isPositive: pred.prediction === 'CONFIRMED'
  }));
};
```

## Testing the Integration

### Step 1: Start Backend
```bash
cd backend
python app.py
```

### Step 2: Start Frontend
```bash
cd frontend
npm run dev
```

### Step 3: Test Flow
1. Go to http://localhost:5173/custom
2. Select dataset and model
3. Set hyperparameters
4. Click "Start Training"
5. Should redirect to progress page
6. Progress bar should update
7. Should redirect to result page
8. Results should display

## Troubleshooting

### CORS Issues
If frontend can't reach backend:
- Check Flask-CORS is installed
- Verify CORS is enabled in app.py
- Check browser console for CORS errors

### File Upload Issues
- Check MAX_CONTENT_LENGTH in config
- Verify file types are allowed
- Check upload folder permissions

### Session Not Found
- Verify session_id is passed correctly
- Check session hasn't timed out
- Verify session is created before querying

## Next Steps

1. Implement background job processing (Celery/RQ)
2. Add database for session persistence
3. Implement user authentication
4. Add rate limiting
5. Add comprehensive error handling
6. Add logging and monitoring
7. Write unit tests
8. Write integration tests

## Resources

- Flask Documentation: https://flask.palletsprojects.com/
- React Documentation: https://react.dev/
- Flask-CORS: https://flask-cors.readthedocs.io/
