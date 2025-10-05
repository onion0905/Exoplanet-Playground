#!/usr/bin/env python3
"""
Test script for frontend-backend integration
"""

import sys
import os
import subprocess
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def start_backend():
    """Start the backend server"""
    print("🚀 Starting backend server...")
    os.chdir(os.path.join(project_root, 'backend'))
    
    # Use subprocess.Popen to run in background
    process = subprocess.Popen(
        [sys.executable, 'app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for server to start
    time.sleep(3)
    
    return process

def start_frontend():
    """Start the frontend development server"""
    print("🎨 Starting frontend server...")
    os.chdir(os.path.join(project_root, 'frontend'))
    
    process = subprocess.Popen(
        ['npm', 'run', 'dev'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    
    # Wait for frontend to start
    time.sleep(5)
    
    return process

def test_backend_endpoints():
    """Test backend API endpoints"""
    print("🔍 Testing backend endpoints...")
    
    base_url = "http://localhost:5000"
    
    tests = [
        {
            'name': 'Health Check',
            'url': f'{base_url}/health',
            'method': 'GET'
        },
        {
            'name': 'Training Start (NASA Kepler)',
            'url': f'{base_url}/api/training/start',
            'method': 'POST',
            'data': {
                'model_type': 'rf',
                'dataset_source': 'nasa',
                'dataset_name': 'kepler',
                'target_column': 'koi_disposition',
                'hyperparameters': {
                    'n_estimators': 10,  # Small for quick test
                    'max_depth': 5
                }
            }
        }
    ]
    
    results = []
    
    for test in tests:
        try:
            print(f"  Testing: {test['name']}...")
            
            if test['method'] == 'GET':
                response = requests.get(test['url'], timeout=10)
            else:
                response = requests.post(
                    test['url'],
                    json=test['data'],
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
            
            if response.status_code < 400:
                print(f"  ✅ {test['name']}: {response.status_code}")
                
                # For training start, get session ID and test progress
                if 'training/start' in test['url']:
                    try:
                        result_data = response.json()
                        session_id = result_data.get('session_id')
                        if session_id:
                            print(f"  📊 Testing progress tracking for session: {session_id}")
                            
                            # Test progress endpoint
                            for i in range(3):  # Check 3 times
                                time.sleep(2)
                                progress_response = requests.get(
                                    f"{base_url}/api/training/progress?session_id={session_id}",
                                    timeout=10
                                )
                                
                                if progress_response.status_code == 200:
                                    progress_data = progress_response.json()
                                    print(f"    Progress {i+1}: {progress_data.get('progress', 0)}% - {progress_data.get('current_step', 'N/A')}")
                                    
                                    if progress_data.get('completed'):
                                        print("    ✅ Training completed!")
                                        
                                        # Test results endpoint
                                        results_response = requests.get(
                                            f"{base_url}/api/training/results?session_id={session_id}",
                                            timeout=10
                                        )
                                        if results_response.status_code == 200:
                                            results_data = results_response.json()
                                            print(f"    📈 Results available - Metrics: {results_data.get('metrics', {})}")
                                        break
                    except Exception as e:
                        print(f"    ⚠️ Progress tracking error: {e}")
                        
                results.append({'test': test['name'], 'status': 'PASS', 'code': response.status_code})
            else:
                print(f"  ❌ {test['name']}: {response.status_code} - {response.text[:100]}")
                results.append({'test': test['name'], 'status': 'FAIL', 'code': response.status_code})
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ {test['name']}: Connection error - {e}")
            results.append({'test': test['name'], 'status': 'ERROR', 'error': str(e)})
        except Exception as e:
            print(f"  ❌ {test['name']}: {e}")
            results.append({'test': test['name'], 'status': 'ERROR', 'error': str(e)})
    
    return results

def test_frontend():
    """Test if frontend is accessible"""
    print("🌐 Testing frontend accessibility...")
    
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print("  ✅ Frontend accessible")
            return True
        else:
            print(f"  ❌ Frontend returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Frontend not accessible: {e}")
        return False

def main():
    """Main integration test"""
    print("🔧 Exoplanet Playground - Frontend-Backend Integration Test")
    print("=" * 60)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        
        # Test backend
        backend_results = test_backend_endpoints()
        
        # Start frontend 
        # Note: This requires npm to be installed and dependencies installed
        try:
            frontend_process = start_frontend()
            frontend_ok = test_frontend()
        except Exception as e:
            print(f"⚠️ Frontend setup issue: {e}")
            print("   Make sure to run 'npm install' in the frontend directory")
            frontend_ok = False
        
        # Summary
        print("\n📊 Integration Test Summary")
        print("=" * 60)
        
        print("Backend API Tests:")
        for result in backend_results:
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"  {status_icon} {result['test']}: {result['status']}")
        
        print(f"\nFrontend: {'✅ OK' if frontend_ok else '❌ FAIL'}")
        
        backend_pass = len([r for r in backend_results if r['status'] == 'PASS'])
        backend_total = len(backend_results)
        
        print(f"\nOverall: {backend_pass}/{backend_total} backend tests passed")
        
        if backend_pass == backend_total and frontend_ok:
            print("🎉 Integration test SUCCESSFUL!")
            return 0
        else:
            print("⚠️ Some issues found. Check logs above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if backend_process:
            backend_process.terminate()
            print("  Backend process terminated")
        if frontend_process:
            frontend_process.terminate()
            print("  Frontend process terminated")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)