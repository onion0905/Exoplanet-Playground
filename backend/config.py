"""
Backend configuration for Exoplanet Playground.
"""

from pathlib import Path

# Base paths
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
UPLOAD_DIR = PROJECT_ROOT / 'uploads'
ML_DIR = PROJECT_ROOT / 'ML'

# Server configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# File upload configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Session configuration
SESSION_TIMEOUT_MINUTES = 60
MAX_CONCURRENT_SESSIONS = 10

# ML configuration
DEFAULT_MODEL_TYPE = 'random_forest'
AVAILABLE_DATASETS = ['kepler', 'tess', 'k2']
AVAILABLE_MODELS = [
    'random_forest',
    'decision_tree', 
    'xgboost',
    'svm',
    'linear_regression',
    'deep_learning',
    'pca'
]

# Pretrained model configuration
PRETRAINED_MODEL_NAME = 'pretrained_random_forest_kepler'
PRETRAINED_MODEL_PATH = ML_DIR / 'models' / 'pretrained'

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
