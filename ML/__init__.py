"""
ML Package for Exoplanet Detection System

This package contains the machine learning components moved from NASA-Hackathon-Fake:
- src/: Source code including APIs, models, data processing, and utilities  
- models/: Trained model storage (pretrained and user models)

Main APIs:
- ExoplanetMLAPI: User-friendly unified interface
- TrainingAPI: Advanced training controls
- PredictionAPI: Model prediction and inference
- ExplanationAPI: Model interpretability and explainability

Usage:
    from ML.src.api.user_api import ExoplanetMLAPI
    
    api = ExoplanetMLAPI()
    datasets = api.list_available_datasets()
    models = api.list_available_models()
"""

__version__ = "1.0.0"
__author__ = "NASA Hackathon Team"