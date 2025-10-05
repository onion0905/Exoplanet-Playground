"""
Route blueprint registration
"""
from flask import Blueprint

# Import all blueprint modules
from .home import home_bp
from .select import select_bp  
from .training import training_bp
from .results import results_bp


def register_blueprints(app):
    """Register all blueprints with the Flask application"""
    
    # Register blueprints with API prefix
    api_prefix = app.config.get('API_PREFIX', '/api')
    
    app.register_blueprint(home_bp, url_prefix=api_prefix)
    app.register_blueprint(select_bp, url_prefix=api_prefix)
    app.register_blueprint(training_bp, url_prefix=api_prefix)
    app.register_blueprint(results_bp, url_prefix=api_prefix)