"""
ML Service - Wrapper for ML operations
"""
import sys
from pathlib import Path
import logging

# Add ML module to path
project_root = Path(__file__).parent.parent.parent
ml_path = project_root / "ML"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(ml_path))

from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI


class MLService:
    """Service class that wraps ML API operations for web endpoints with three-class prediction support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._user_api = None
        self._training_api = None
        self._prediction_api = None
        self._explanation_api = None
    
    @property
    def user_api(self):
        """Lazy initialization of user API"""
        if self._user_api is None:
            self._user_api = ExoplanetMLAPI()
        return self._user_api
    
    @property
    def training_api(self):
        """Lazy initialization of training API"""
        if self._training_api is None:
            self._training_api = TrainingAPI()
        return self._training_api
    
    @property
    def prediction_api(self):
        """Lazy initialization of prediction API"""
        if self._prediction_api is None:
            self._prediction_api = PredictionAPI()
        return self._prediction_api
    
    @property
    def explanation_api(self):
        """Lazy initialization of explanation API"""
        if self._explanation_api is None:
            self._explanation_api = ExplanationAPI(prediction_api=self.prediction_api)
        return self._explanation_api
    
    # Dataset operations
    def get_available_datasets(self):
        """Get list of available NASA datasets"""
        try:
            return self.user_api.list_available_datasets()
        except Exception as e:
            self.logger.error(f"Error getting datasets: {e}")
            raise
    
    def get_available_models(self):
        """Get list of available model types"""
        try:
            return self.user_api.list_available_models()
        except Exception as e:
            self.logger.error(f"Error getting models: {e}")
            raise
    
    def get_trained_models(self):
        """Get list of trained models"""
        try:
            return self.user_api.list_trained_models()
        except Exception as e:
            self.logger.error(f"Error getting trained models: {e}")
            raise
    
    # Training operations
    def start_training_session(self, session_id: str):
        """Start a new training session"""
        try:
            return self.training_api.start_training_session(session_id)
        except Exception as e:
            self.logger.error(f"Error starting training session {session_id}: {e}")
            raise
    
    def load_data_for_training(self, session_id: str, data_source: str, data_config: dict):
        """Load data for training"""
        try:
            return self.training_api.load_data_for_training(session_id, data_source, data_config)
        except Exception as e:
            self.logger.error(f"Error loading data for session {session_id}: {e}")
            raise
    
    def configure_training(self, session_id: str, training_config: dict):
        """Configure training parameters"""
        try:
            return self.training_api.configure_training(session_id, training_config)
        except Exception as e:
            self.logger.error(f"Error configuring training for session {session_id}: {e}")
            raise
    
    def start_training(self, session_id: str):
        """Start the actual training process"""
        try:
            return self.training_api.start_training(session_id)
        except Exception as e:
            self.logger.error(f"Error starting training for session {session_id}: {e}")
            raise
    
    def get_training_progress(self, session_id: str):
        """Get training progress"""
        try:
            return self.training_api.get_session_info(session_id)
        except Exception as e:
            self.logger.error(f"Error getting training progress for session {session_id}: {e}")
            raise
    
    def save_trained_model(self, session_id: str, model_name: str):
        """Save trained model"""
        try:
            return self.training_api.save_trained_model(session_id, model_name)
        except Exception as e:
            self.logger.error(f"Error saving model for session {session_id}: {e}")
            raise
    
    # Prediction operations
    def load_model(self, model_path: str, model_id: str = None):
        """Load a model for predictions"""
        try:
            return self.prediction_api.load_model(model_path, model_id)
        except Exception as e:
            self.logger.error(f"Error loading model {model_path}: {e}")
            raise
    
    def predict_single(self, model_id: str, input_data: dict):
        """Make single prediction"""
        try:
            return self.prediction_api.predict_single(model_id, input_data)
        except Exception as e:
            self.logger.error(f"Error making prediction with model {model_id}: {e}")
            raise
    
    def predict_batch(self, model_id: str, input_data: list):
        """Make batch predictions"""
        try:
            return self.prediction_api.predict_batch(model_id, input_data)
        except Exception as e:
            self.logger.error(f"Error making batch predictions with model {model_id}: {e}")
            raise
    
    def get_prediction_confidence(self, model_id: str, input_data: dict):
        """Get prediction with confidence"""
        try:
            return self.prediction_api.get_prediction_confidence(model_id, input_data)
        except Exception as e:
            self.logger.error(f"Error getting prediction confidence with model {model_id}: {e}")
            raise
    
    # Explanation operations
    def explain_model_global(self, model_id: str, X_train, y_train, X_test, y_test):
        """Get global model explanations"""
        try:
            return self.explanation_api.explain_model_global(model_id, X_train, y_train, X_test, y_test)
        except Exception as e:
            self.logger.error(f"Error explaining model {model_id}: {e}")
            raise
    
    def explain_prediction_local(self, model_id: str, instance_data: dict, reference_data=None):
        """Get local prediction explanation"""
        try:
            return self.explanation_api.explain_prediction_local(model_id, instance_data, reference_data)
        except Exception as e:
            self.logger.error(f"Error explaining prediction for model {model_id}: {e}")
            raise


# Global service instance
ml_service = MLService()