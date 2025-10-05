"""
Home page routes - Project information endpoints
"""
import time
from flask import Blueprint, jsonify

home_bp = Blueprint('home', __name__)


@home_bp.route('/')
def get_project_info():
    """Get project information for the homepage"""
    project_info = {
        'name': 'NASA Exoplanet Machine Learning Playground',
        'description': 'An interactive web application for training custom machine learning models to identify exoplanets using NASA datasets.',
        'version': '1.0.0',
        'features': [
            'Multiple NASA datasets (Kepler, TESS, K2)',
            'Various ML algorithms (Random Forest, SVM, XGBoost, etc.)',
            'Custom data upload and preprocessing',
            'Real-time training progress monitoring',
            'Interactive prediction results',
            'Model explainability and feature importance analysis',
            'Data visualization tools'
        ],
        'datasets': {
            'kepler': {
                'name': 'Kepler Objects of Interest',
                'description': 'Data from NASA\'s Kepler Space Telescope mission',
                'features': 'Stellar and planetary parameters from Kepler observations',
                'size': '~10,000 candidates'
            },
            'tess': {
                'name': 'TESS Objects of Interest',
                'description': 'Data from NASA\'s Transiting Exoplanet Survey Satellite',
                'features': 'Transit photometry and stellar characterization',
                'size': '~5,000 candidates'
            },
            'k2': {
                'name': 'K2 Mission Data',
                'description': 'Extended mission data from Kepler\'s K2 campaign',
                'features': 'Additional exoplanet candidates and confirmations',
                'size': '~2,000 candidates'
            }
        },
        'models': [
            'Random Forest',
            'Decision Tree', 
            'Linear Regression',
            'Support Vector Machine (SVM)',
            'XGBoost',
            'Principal Component Analysis (PCA)',
            'Deep Learning (Neural Networks)'
        ],
        'workflow': [
            'Select or upload your dataset',
            'Choose a machine learning model',
            'Configure hyperparameters',
            'Monitor real-time training progress',
            'Analyze results and predictions',
            'Explore model explanations and feature importance'
        ],
        'api_endpoints': {
            'datasets': '/api/datasets',
            'models': '/api/models', 
            'training': '/api/training',
            'predictions': '/api/predictions',
            'explanations': '/api/explanations'
        }
    }
    
    return jsonify(project_info)


@home_bp.route('/status')
def get_system_status():
    """Get system status and health information"""
    try:
        from services import ml_service
        
        # Test ML service connectivity
        datasets = ml_service.get_available_datasets()
        models = ml_service.get_available_models()
        
        status = {
            'status': 'healthy',
            'ml_service': 'connected',
            'available_datasets': len(datasets),
            'available_models': len(models),
            'timestamp': int(time.time())
        }
        
        return jsonify(status)
        
    except Exception as e:
        status = {
            'status': 'degraded',
            'ml_service': 'disconnected',
            'error': str(e),
            'timestamp': int(time.time())
        }
        
        return jsonify(status), 503


@home_bp.route('/about')
def get_about_info():
    """Get detailed about information"""
    about_info = {
        'mission': 'To democratize exoplanet discovery by providing accessible machine learning tools for researchers, students, and citizen scientists.',
        'technology': {
            'frontend': 'React + Vite + TailwindCSS',
            'backend': 'Flask + Python',
            'ml_framework': 'scikit-learn, XGBoost, TensorFlow',
            'data_processing': 'pandas, numpy'
        },
        'data_sources': [
            'NASA Exoplanet Archive',
            'Kepler/K2 Mission Data',
            'TESS Mission Data'
        ],
        'team': 'NASA Hackathon 2024 Team',
        'github': 'https://github.com/onion0905/Exoplanet-Playground',
        'license': 'MIT License'
    }
    
    return jsonify(about_info)