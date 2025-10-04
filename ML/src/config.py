import os
import numpy as np
from pathlib import Path


# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"  # Root data directory for CSV files
MODELS_DIR = Path(__file__).parent.parent / "models"  # ML/models directory
SRC_DIR = Path(__file__).parent  # ML/src directory

# Data configuration
NASA_DATASETS = {
    'kepler': 'kepler_objects_of_interest.csv',
    'tess': 'tess_objects_of_interest.csv', 
    'k2': 'k2_planets_and_candidates.csv'
}

# Model configuration
MODEL_SAVE_DIR = MODELS_DIR / "user"
PRETRAINED_MODEL_DIR = MODELS_DIR / "pretrained"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# API configuration
API_CONFIG = {
    'max_file_size_mb': 100,
    'allowed_file_extensions': ['.csv', '.xlsx', '.xls', '.json'],
    'session_timeout_minutes': 60,
    'max_concurrent_sessions': 10
}

# Model training configuration
TRAINING_CONFIG = {
    'default_test_size': 0.2,
    'default_validation_size': 0.1,
    'random_state': 42,
    'max_training_time_minutes': 30,
    'early_stopping_patience': 10
}

# Feature selection configuration
FEATURE_CONFIG = {
    'max_missing_ratio': 0.5,
    'min_variance_threshold': 0.01,
    'correlation_threshold': 0.95,
    'max_features_for_exhaustive_search': 20
}

# Explainability configuration
EXPLAINABILITY_CONFIG = {
    'max_features_to_analyze': 50,
    'permutation_importance_repeats': 5,
    'feature_importance_methods': [
        'model_importance',
        'column_drop',
        'average_replacement', 
        'permutation'
    ]
}

# Target variable mappings for common exoplanet datasets
TARGET_MAPPINGS = {
    'koi_disposition': {
        'CONFIRMED': 'planet',
        'CANDIDATE': 'candidate',
        'FALSE POSITIVE': 'false_positive',
        'NOT DISPOSITIONED': 'unknown'
    },
    'disposition': {
        'Confirmed': 'planet',
        'Candidate': 'candidate', 
        'False Positive': 'false_positive'
    },
    'tfopwg_disp': {
        'PC': 'planet_candidate',
        'FP': 'false_positive',
        'KP': 'known_planet'
    }
}

# Model hyperparameter ranges for validation
HYPERPARAMETER_RANGES = {
    'linear_regression': {
        'C': (0.001, 100.0),
        'max_iter': (100, 5000)
    },
    'svm': {
        'C': (0.001, 100.0),
        'gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, 6))
    },
    'decision_tree': {
        'max_depth': (1, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 20)
    },
    'random_forest': {
        'n_estimators': (10, 500),
        'max_depth': (1, 50),
        'min_samples_split': (2, 20)
    },
    'xgboost': {
        'n_estimators': (10, 500),
        'max_depth': (1, 20),
        'learning_rate': (0.01, 0.3)
    },
    'pca': {
        'n_components': (0.5, 0.99),
        'C': (0.001, 100.0)
    },
    'deep_learning': {
        'learning_rate': (0.0001, 0.1),
        'dropout_rate': (0.0, 0.8),
        'hidden_layers': [[32], [64], [128], [64, 32], [128, 64], [128, 64, 32]]
    }
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'accuracy': {
        'excellent': 0.9,
        'good': 0.8,
        'fair': 0.7,
        'poor': 0.0
    },
    'f1_score': {
        'excellent': 0.85,
        'good': 0.75,
        'fair': 0.65,
        'poor': 0.0
    }
}

# Import numpy for hyperparameter ranges
try:
    import numpy as np
except ImportError:
    # Fallback if numpy not available
    HYPERPARAMETER_RANGES['svm']['gamma'] = ['scale', 'auto']