"""
Configuration management for the backend application
"""
import os
from pathlib import Path

# Base directory
basedir = Path(__file__).parent.parent.absolute()


class BaseConfig:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'exoplanet-ml-secret-key-dev')
    DEBUG = False
    TESTING = False
    
    # ML Configuration
    ML_MODEL_DIR = basedir / 'ML' / 'models'
    DATA_DIR = basedir / 'data'
    UPLOAD_FOLDER = basedir / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    
    # API Configuration
    API_PREFIX = '/api'
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    MAX_CONCURRENT_SESSIONS = 10


class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    DEBUG = True
    ENV = 'development'


class ProductionConfig(BaseConfig):
    """Production configuration"""
    DEBUG = False
    ENV = 'production'
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")


class TestingConfig(BaseConfig):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    ENV = 'testing'


config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name='development'):
    """Get configuration class by name"""
    return config_map.get(config_name, config_map['default'])