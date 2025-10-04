"""
Exoplanet ML Training System

A comprehensive machine learning system for training and deploying 
exoplanet classification models using NASA datasets.

This system provides:
- Multiple ML algorithms (Linear Regression, SVM, Decision Tree, Random Forest, XGBoost, PCA, Deep Learning)
- Data loading and processing for NASA exoplanet datasets
- Feature importance analysis and explainability tools
- Column selection and dropping functionality  
- Training, prediction, and explanation APIs
- Model validation and performance metrics

Usage:
    from ML.src.api.training_api import TrainingAPI
    from ML.src.api.prediction_api import PredictionAPI
    from ML.src.api.explanation_api import ExplanationAPI
    
    # Training
    training_api = TrainingAPI()
    session_id = "my_session"
    training_api.start_training_session(session_id)
    
    # Prediction
    prediction_api = PredictionAPI()
    prediction_api.load_model("path/to/model")
    
    # Explanation
    explanation_api = ExplanationAPI()
    explanation_api.explain_model_global(...)
"""

__version__ = "1.0.0"
__author__ = "NASA Hackathon Team"

# Import main API classes for easy access
from .api.training_api import TrainingAPI
from .api.prediction_api import PredictionAPI
from .api.explanation_api import ExplanationAPI

# Import model factory
from .utils.model_factory import ModelFactory

# Import data utilities
from .data.data_loader import DataLoader
from .data.data_processor import DataProcessor
from .data.data_validator import DataValidator

__all__ = [
    'TrainingAPI',
    'PredictionAPI', 
    'ExplanationAPI',
    'ModelFactory',
    'DataLoader',
    'DataProcessor',
    'DataValidator'
]