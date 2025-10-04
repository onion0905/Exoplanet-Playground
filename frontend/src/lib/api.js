// API utility for communicating with Flask backend
const API_BASE_URL = 'http://localhost:5000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Homepage data
  async getHomeData() {
    return this.request('/');
  }

  // Available datasets
  async getDatasets() {
    return this.request('/api/datasets');
  }

  // Available models
  async getModels() {
    return this.request('/api/models');
  }

  // Health check
  async getHealth() {
    return this.request('/api/health');
  }

  // Frontend status
  async getFrontendStatus() {
    return this.request('/api/frontend-status');
  }

  // Session management
  async createSession() {
    return this.request('/select');
  }

  async getSessionInfo(sessionId) {
    return this.request(`/select?session_id=${sessionId}`);
  }

  async updateDataSource(sessionId, dataSource, dataset = null, uploadedFileInfo = null) {
    return this.request('/select', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        action: 'update_data_source',
        data_source: dataSource,
        dataset: dataset,
        uploaded_file_info: uploadedFileInfo
      })
    });
  }

  async updateModel(sessionId, modelType) {
    return this.request('/select', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        action: 'update_model',
        model_type: modelType
      })
    });
  }

  async updateHyperparameters(sessionId, hyperparameters, preprocessingConfig = {}) {
    return this.request('/select', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        action: 'update_hyperparameters',
        hyperparameters: hyperparameters,
        preprocessing_config: preprocessingConfig
      })
    });
  }

  async nextStep(sessionId) {
    return this.request('/select', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        action: 'next_step'
      })
    });
  }

  async setStep(sessionId, step) {
    return this.request('/select', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        action: 'set_step',
        step: step
      })
    });
  }

  // Get model hyperparameters configuration
  async getModelHyperparameters(modelType) {
    return this.request(`/api/hyperparameters/${modelType}`);
  }

  // Training
  async getTrainingProgress(sessionId) {
    return this.request(`/training?session_id=${sessionId}`);
  }

  async startTraining(sessionId) {
    return this.request('/training', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId })
    });
  }

  // Predictions
  async getPredictionStatus(sessionId) {
    return this.request(`/predict?session_id=${sessionId}`);
  }

  async submitTestData(sessionId, testDataConfig) {
    return this.request('/predict', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        test_data_config: testDataConfig
      })
    });
  }

  // Results
  async getResults(sessionId) {
    return this.request(`/result?session_id=${sessionId}`);
  }

  async generateExplanation(sessionId, instanceData, explanationType = 'local') {
    return this.request('/result', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        instance_data: instanceData,
        explanation_type: explanationType
      })
    });
  }

  // File upload
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const url = `${this.baseURL}/api/upload`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed! status: ${response.status}`);
    }

    return await response.json();
  }

  // Upload CSV for training data
  async uploadCSV(file, sessionId = null) {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const url = `${this.baseURL}/api/upload-csv`;
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Upload failed with status ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('CSV Upload Error:', error);
      throw error;
    }
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;