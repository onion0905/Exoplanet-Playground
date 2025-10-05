"""
Middleware setup for the Flask application
"""
import logging
from flask import request, g
from werkzeug.exceptions import HTTPException
import time


def setup_middleware(app):
    """Setup all middleware for the application"""
    
    # Request logging middleware
    @app.before_request
    def log_request():
        g.start_time = time.time()
        app.logger.info(f'{request.method} {request.path} - {request.remote_addr}')
    
    @app.after_request
    def log_response(response):
        duration = time.time() - getattr(g, 'start_time', time.time())
        app.logger.info(f'{request.method} {request.path} - {response.status_code} - {duration:.3f}s')
        return response
    
    # Error handling middleware
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found', 'message': str(error)}, 404
    
    @app.errorhandler(400)
    def bad_request(error):
        return {'error': 'Bad request', 'message': str(error)}, 400
    
    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f'Internal server error: {error}')
        return {'error': 'Internal server error', 'message': 'An unexpected error occurred'}, 500
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        return {
            'error': error.name,
            'message': error.description,
            'code': error.code
        }, error.code
    
    # General exception handler
    @app.errorhandler(Exception)
    def handle_exception(error):
        app.logger.error(f'Unhandled exception: {error}', exc_info=True)
        return {
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }, 500