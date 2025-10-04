#!/usr/bin/env python3
"""
Backend API Server for Exoplanet Playground
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.app import app
from backend.config import config

def main():
    """Main entry point for the backend API server."""
    
    # Get environment
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Configure app
    app.config.from_object(config.get(env, config['default']))
    
    # Print startup information
    print("🌌 Exoplanet Discovery Playground - Backend API Server")
    print("=" * 65)
    print(f"Environment: {env}")
    print(f"Debug mode: {app.config['DEBUG']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f}MB")
    print("=" * 65)
    
    print("🎨 Development Setup:")
    print("   🔧 Backend API Server (This terminal)")
    print("   🌐 Frontend: cd frontend && npm run dev")
    print("   📖 Setup guide: DEVELOPMENT_SETUP.md")
    
    print("=" * 65)
    print("🚀 Starting Flask API server...")
    print("📡 Available API endpoints:")
    print("   🏠 GET  /                    - Homepage data (JSON)")
    print("   🎯 GET  /select             - Model selection API")
    print("   📊 GET  /training           - Training progress API")  
    print("   🔍 GET  /predict            - Prediction API")
    print("   📈 GET  /result             - Results API")
    print("   🔧 GET  /api/health         - System health check")
    print("   🗂️  GET  /api/datasets       - Available datasets info")
    print("   🤖 GET  /api/models         - Available ML models info")
    print("   ⚡ POST /api/upload         - File upload endpoint")
    print("=" * 65)
    print("🌍 Backend API: http://localhost:5000")
    print("🎨 Frontend App: http://localhost:3000 (run: npm run dev)")
    print("=" * 65)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config['DEBUG']
    )

if __name__ == '__main__':
    main()