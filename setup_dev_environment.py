#!/usr/bin/env python3
"""
Setup script for Exoplanet Playground development environment
"""

import os
import sys
import subprocess
import json

def run_command(command, cwd=None, description=None):
    """Run a command and handle errors"""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            if description:
                print(f"  ✅ {description} completed")
            return True
        else:
            print(f"  ❌ Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout: {description} took too long")
        return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking requirements...")
    
    requirements = [
        ('python', 'python --version'),
        ('pip', 'pip --version'),
        ('node', 'node --version'),
        ('npm', 'npm --version')
    ]
    
    missing = []
    
    for name, command in requirements:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print(f"  ✅ {name}: {version}")
            else:
                missing.append(name)
                print(f"  ❌ {name}: Not found")
        except Exception:
            missing.append(name)
            print(f"  ❌ {name}: Not found")
    
    return missing

def setup_backend():
    """Set up the backend environment"""
    print("\n🐍 Setting up backend...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(project_root, 'backend')
    
    # Install Python requirements
    requirements_file = os.path.join(project_root, 'requirements.txt')
    
    if os.path.exists(requirements_file):
        success = run_command(
            f'pip install -r "{requirements_file}"',
            description="Installing Python dependencies"
        )
        if not success:
            print("  ⚠️ Failed to install some Python packages")
            return False
    else:
        print(f"  ⚠️ Requirements file not found: {requirements_file}")
    
    # Create necessary directories
    dirs_to_create = [
        os.path.join(project_root, 'uploads'),
        os.path.join(project_root, 'ML', 'models', 'user'),
        os.path.join(project_root, 'ML', 'models', 'pretrained')
    ]
    
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"  📁 Created directory: {dir_path}")
    
    print("  ✅ Backend setup completed")
    return True

def setup_frontend():
    """Set up the frontend environment"""
    print("\n🎨 Setting up frontend...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_dir = os.path.join(project_root, 'frontend')
    
    if not os.path.exists(frontend_dir):
        print(f"  ❌ Frontend directory not found: {frontend_dir}")
        return False
    
    # Check if package.json exists
    package_json = os.path.join(frontend_dir, 'package.json')
    if not os.path.exists(package_json):
        print(f"  ❌ package.json not found in {frontend_dir}")
        return False
    
    # Install npm dependencies
    success = run_command(
        'npm install',
        cwd=frontend_dir,
        description="Installing Node.js dependencies"
    )
    
    if success:
        print("  ✅ Frontend setup completed")
        return True
    else:
        print("  ❌ Frontend setup failed")
        return False

def create_env_file():
    """Create environment configuration file"""
    print("\n📝 Creating environment configuration...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(project_root, 'backend')
    
    env_content = """# Exoplanet Playground Environment Configuration
FLASK_APP=app.py
FLASK_ENV=development
DEBUG=True

# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=5000

# Frontend Configuration (for CORS)
FRONTEND_URL=http://localhost:3000

# Data paths
DATA_DIR=../data
MODELS_DIR=../ML/models
UPLOADS_DIR=../uploads

# ML Configuration
DEFAULT_MODEL_TYPE=rf
MAX_TRAINING_TIME=300

# Session Configuration
SESSION_TIMEOUT=3600
"""
    
    env_file = os.path.join(backend_dir, '.env')
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"  ✅ Created {env_file}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create .env file: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("\n🚀 Setup Complete! Here's how to use the application:")
    print("=" * 60)
    
    instructions = """
1. Start the Backend Server:
   cd backend
   python app.py
   
   The backend will be available at: http://localhost:5000

2. Start the Frontend Server (in a new terminal):
   cd frontend
   npm run dev
   
   The frontend will be available at: http://localhost:3000

3. Access the Application:
   Open your browser and go to: http://localhost:3000

4. Test the Integration:
   python scripts/integration/test_frontend_backend.py

API Endpoints:
- Health Check: GET http://localhost:5000/health
- Start Training: POST http://localhost:5000/api/training/start
- Check Progress: GET http://localhost:5000/api/training/progress?session_id=<id>
- Get Results: GET http://localhost:5000/api/training/results?session_id=<id>
- Upload File: POST http://localhost:5000/api/upload

Troubleshooting:
- If ports are in use, modify the configuration in vite.config.js and backend/config/
- Check console logs for detailed error messages
- Ensure all dependencies are installed correctly
"""
    
    print(instructions)

def main():
    """Main setup function"""
    print("🛠️  Exoplanet Playground Setup")
    print("=" * 60)
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        print("Please install the missing tools before continuing.")
        return 1
    
    # Setup backend
    backend_ok = setup_backend()
    
    # Setup frontend
    frontend_ok = setup_frontend()
    
    # Create environment file
    env_ok = create_env_file()
    
    # Summary
    print(f"\n📊 Setup Summary")
    print("=" * 60)
    print(f"Backend: {'✅ OK' if backend_ok else '❌ FAIL'}")
    print(f"Frontend: {'✅ OK' if frontend_ok else '❌ FAIL'}")
    print(f"Environment: {'✅ OK' if env_ok else '❌ FAIL'}")
    
    if backend_ok and frontend_ok and env_ok:
        print("\n🎉 Setup completed successfully!")
        show_usage_instructions()
        return 0
    else:
        print("\n⚠️ Setup completed with issues. Check the logs above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)