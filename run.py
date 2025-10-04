#!/usr/bin/env python3
"""
Startup script for Exoplanet Playground Backend
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
    """Main entry point for the application."""
    
    # Get environment
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Configure app
    app.config.from_object(config.get(env, config['default']))
    
    # Check frontend build status
    frontend_dist = project_root / 'frontend' / 'dist'
    frontend_ready = frontend_dist.exists() and (frontend_dist / 'index.html').exists()
    
    # Print startup information
    print("🌌 Exoplanet Discovery Playground Backend")
    print("=" * 60)
    print(f"Environment: {env}")
    print(f"Debug mode: {app.config['DEBUG']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f}MB")
    print("=" * 60)
    
    # Frontend status
    print("🎨 Frontend Integration Status:")
    if frontend_ready:
        print("   ✅ React frontend built and ready")
        print(f"   📂 Serving from: {frontend_dist}")
        print("   🌐 Access full app at: http://localhost:5000/")
    else:
        print("   ⚠️  Frontend not built - will show build instructions")
        print("   � Run: cd frontend && npm run build")
    
    print("=" * 60)
    print("�🚀 Starting server...")
    print("📡 Available endpoints:")
    
    if frontend_ready:
        print("   🏠 GET  /                    - Homepage (React app or fallback)")
        print("   🎯 GET  /select             - Model selection (React route)")
        print("   📊 GET  /training           - Training progress (React route)")
        print("   🔍 GET  /predict            - Prediction interface (React route)")
        print("   📈 GET  /result             - Results visualization (React route)")
        print("   📚 GET  /learn              - Learning resources (React route)")
    else:
        print("   🏠 GET  /                    - Build instructions page")
    
    print("   🔧 GET  /api/health         - System health check")
    print("   📋 GET  /api/frontend-status - Frontend build status") 
    print("   🗂️  GET  /api/datasets       - Available datasets info")
    print("   🤖 GET  /api/models         - Available ML models info")
    print("   ⚡ POST /api/upload         - File upload endpoint")
    print("=" * 60)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config['DEBUG']
    )

if __name__ == '__main__':
    main()