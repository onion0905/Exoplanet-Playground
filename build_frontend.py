#!/usr/bin/env python3
"""
Build script for Exoplanet Playground Frontend
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return success status."""
    try:
        print(f"Running: {command}")
        if cwd:
            print(f"In directory: {cwd}")
        
        result = subprocess.run(
            command, 
            shell=shell, 
            cwd=cwd, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ Failed!")
            if result.stderr:
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    """Build the frontend and optionally start the server."""
    
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    
    print("🌌 Exoplanet Playground Build Script")
    print("=" * 50)
    
    # Check if frontend directory exists
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        if not run_command("npm install", cwd=str(frontend_dir)):
            print("❌ Failed to install dependencies!")
            return False
    
    # Build frontend
    print("🔨 Building frontend...")
    if not run_command("npm run build", cwd=str(frontend_dir)):
        print("❌ Frontend build failed!")
        return False
    
    # Check if build was successful
    dist_dir = frontend_dir / "dist"
    if not dist_dir.exists():
        print("❌ Build directory not created!")
        return False
    
    print("✅ Frontend built successfully!")
    
    # Ask if user wants to start the server
    start_server = input("\n🚀 Start the backend server? (y/n): ").lower().strip()
    
    if start_server in ['y', 'yes']:
        print("\n🔧 Starting backend server...")
        
        # Check if virtual environment exists
        venv_path = project_root / ".venv"
        if venv_path.exists():
            if sys.platform == "win32":
                python_exe = str(venv_path / "Scripts" / "python.exe")
            else:
                python_exe = str(venv_path / "bin" / "python")
        else:
            python_exe = "python"
        
        # Start server
        server_command = f'"{python_exe}" run.py'
        print(f"\nStarting server with: {server_command}")
        print("🌍 Server will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)
        
        try:
            subprocess.run(server_command, shell=True, cwd=str(project_root))
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
