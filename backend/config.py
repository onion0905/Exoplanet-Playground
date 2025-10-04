"""
Configuration file for Exoplanet Playground Backend
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = BASE_DIR / "ML"
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "ML" / "models" / "user"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Flask configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = str(UPLOADS_DIR)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # CORS settings
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]  # React dev servers
    
    # ML Model settings
    MODEL_SAVE_DIR = str(MODELS_DIR)
    
    # NASA datasets configuration
    NASA_DATASETS = {
        'kepler': {
            'file': 'kepler_objects_of_interest.csv',
            'target_column': 'koi_disposition',
            'positive_class': 'CONFIRMED',
            'negative_class': 'FALSE POSITIVE'
        },
        'k2': {
            'file': 'k2_planets_and_candidates.csv', 
            'target_column': 'disposition',
            'positive_class': 'CONFIRMED',
            'negative_class': 'FALSE POSITIVE'
        },
        'tess': {
            'file': 'tess_objects_of_interest.csv',
            'target_column': 'tfopwg_disp',
            'positive_class': 'CP',
            'negative_class': 'FP'
        }
    }

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False

# Default configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}