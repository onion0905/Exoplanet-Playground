"""
Main Flask application for Exoplanet ML Backend
"""
import os
import sys
from pathlib import Path
from flask import Flask
from flask_cors import CORS

# Add the project root and ML module to Python path
project_root = Path(__file__).parent.parent
ml_path = project_root / "ML"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(ml_path))

from config import get_config
from routes import register_blueprints
from middleware import setup_middleware


def create_app(config_name='development'):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:5173"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Setup middleware
    setup_middleware(app)
    
    # Register blueprints
    register_blueprints(app)
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return {'status': 'healthy', 'service': 'exoplanet-ml-backend'}
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=True
    )