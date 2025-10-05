const API_BASE_URL = 'http://localhost:5000/api';

/**
 * Custom Model Training API
 */
export const customTraining = {
  /**
   * Start custom model training
   * @param {Object} config - Training configuration
   * @param {string} config.model_type - Model type (e.g., 'random_forest')
   * @param {string} config.data_source - 'nasa' or 'user'
   * @param {string} config.dataset_name - For NASA data: 'kepler', 'tess', or 'k2'
   * @param {string} config.data_format - For user data: format type
   * @param {File} config.training_file - Training CSV file (for user data)
   * @param {File} config.testing_file - Testing CSV file (for user data)
   * @param {Object} config.hyperparameters - Model hyperparameters
   * @returns {Promise<{success: boolean, session_id: string, message: string}>}
   */
  async startTraining(config) {
    const formData = new FormData();
    
    formData.append('model_type', config.model_type);
    formData.append('data_source', config.data_source);
    
    if (config.data_source === 'nasa') {
      formData.append('dataset_name', config.dataset_name);
    } else {
      formData.append('data_format', config.data_format);
      if (config.training_file) {
        formData.append('training_file', config.training_file);
      }
      if (config.testing_file) {
        formData.append('testing_file', config.testing_file);
      }
    }
    
    if (config.hyperparameters) {
      formData.append('hyperparameters', JSON.stringify(config.hyperparameters));
    }
    
    const response = await fetch(`${API_BASE_URL}/custom/train`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Training failed to start');
    }
    
    return await response.json();
  },

  /**
   * Get training progress
   * @param {string} sessionId - Session ID
   * @returns {Promise<{success: boolean, progress: number, current_step: string, status: string}>}
   */
  async getProgress(sessionId) {
    const response = await fetch(`${API_BASE_URL}/custom/progress/${sessionId}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get progress');
    }
    
    return await response.json();
  },

  /**
   * Get training results
   * @param {string} sessionId - Session ID
   * @returns {Promise<{success: boolean, metrics: Object, predictions: Array, model_info: Object}>}
   */
  async getResult(sessionId) {
    const response = await fetch(`${API_BASE_URL}/custom/result/${sessionId}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get results');
    }
    
    return await response.json();
  }
};

/**
 * Pretrained Model Prediction API
 */
export const pretrainedPrediction = {
  /**
   * Start pretrained model prediction
   * @param {Object} config - Prediction configuration
   * @param {string} config.data_source - 'nasa' or 'user'
   * @param {string} config.dataset_name - For NASA data: 'kepler', 'tess', or 'k2'
   * @param {string} config.data_format - For user data: format type
   * @param {File} config.training_file - Data file for fine-tuning (for user data)
   * @returns {Promise<{success: boolean, session_id: string, message: string}>}
   */
  async startPrediction(config) {
    const formData = new FormData();
    
    formData.append('data_source', config.data_source);
    
    if (config.data_source === 'nasa') {
      formData.append('dataset_name', config.dataset_name);
    } else {
      formData.append('data_format', config.data_format);
      if (config.training_file) {
        formData.append('training_file', config.training_file);
      }
    }
    
    const response = await fetch(`${API_BASE_URL}/pretrained/predict`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Prediction failed to start');
    }
    
    return await response.json();
  },

  /**
   * Get prediction progress
   * @param {string} sessionId - Session ID
   * @returns {Promise<{success: boolean, progress: number, current_step: string, status: string}>}
   */
  async getProgress(sessionId) {
    const response = await fetch(`${API_BASE_URL}/pretrained/progress/${sessionId}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get progress');
    }
    
    return await response.json();
  },

  /**
   * Get prediction results
   * @param {string} sessionId - Session ID
   * @returns {Promise<{success: boolean, metrics: Object, predictions: Array, model_info: Object}>}
   */
  async getResult(sessionId) {
    const response = await fetch(`${API_BASE_URL}/pretrained/result/${sessionId}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get results');
    }
    
    return await response.json();
  }
};

/**
 * Utility API
 */
export const utils = {
  /**
   * Get available datasets
   * @returns {Promise<{success: boolean, datasets: Array<string>}>}
   */
  async getDatasets() {
    const response = await fetch(`${API_BASE_URL}/datasets`);
    return await response.json();
  },

  /**
   * Get available models
   * @returns {Promise<{success: boolean, models: Array<string>}>}
   */
  async getModels() {
    const response = await fetch(`${API_BASE_URL}/models`);
    return await response.json();
  },

  /**
   * Health check
   * @returns {Promise<{success: boolean, status: string, message: string}>}
   */
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return await response.json();
  }
};
