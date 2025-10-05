"""
Quick test script for backend API endpoints.
"""

import requests
import time
import json

BASE_URL = 'http://localhost:5000/api'

def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f'{BASE_URL}/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_datasets():
    """Test datasets endpoint."""
    print("\n=== Testing Datasets Endpoint ===")
    response = requests.get(f'{BASE_URL}/datasets')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_models():
    """Test models endpoint."""
    print("\n=== Testing Models Endpoint ===")
    response = requests.get(f'{BASE_URL}/models')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_custom_training():
    """Test custom training flow."""
    print("\n=== Testing Custom Training Flow ===")
    
    # Start training
    print("\n1. Starting training...")
    payload = {
        'model_type': 'random_forest',
        'data_source': 'nasa',
        'dataset_name': 'kepler',
        'hyperparameters': json.dumps({
            'n_estimators': 10,
            'max_depth': 5
        })
    }
    
    response = requests.post(f'{BASE_URL}/custom/train', data=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    
    if not result.get('success'):
        return False
    
    session_id = result['session_id']
    print(f"Session ID: {session_id}")
    
    # Poll progress
    print("\n2. Polling progress...")
    max_attempts = 60
    for i in range(max_attempts):
        response = requests.get(f'{BASE_URL}/custom/progress/{session_id}')
        progress_data = response.json()
        
        if progress_data.get('success'):
            print(f"Progress: {progress_data['progress']}% - {progress_data['current_step']} ({progress_data['status']})")
            
            if progress_data['status'] == 'completed':
                print("\nTraining completed!")
                break
            elif progress_data['status'] == 'error':
                print(f"\nTraining failed: {progress_data}")
                return False
        
        time.sleep(2)
    else:
        print("\nTimeout waiting for training completion")
        return False
    
    # Get results
    print("\n3. Getting results...")
    response = requests.get(f'{BASE_URL}/custom/result/{session_id}')
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\nMetrics: {result['metrics']}")
        print(f"Model Info: {result['model_info']}")
        print(f"Number of predictions: {len(result['predictions'])}")
        if result['predictions']:
            print(f"Sample prediction: {result['predictions'][0]}")
        return True
    else:
        print(f"Error getting results: {result}")
        return False

def test_pretrained():
    """Test pretrained model flow."""
    print("\n=== Testing Pretrained Model Flow ===")
    
    # Start prediction
    print("\n1. Starting prediction...")
    payload = {
        'data_source': 'nasa',
        'dataset_name': 'kepler'
    }
    
    response = requests.post(f'{BASE_URL}/pretrained/predict', data=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    
    if not result.get('success'):
        return False
    
    session_id = result['session_id']
    print(f"Session ID: {session_id}")
    
    # Poll progress
    print("\n2. Polling progress...")
    max_attempts = 60
    for i in range(max_attempts):
        response = requests.get(f'{BASE_URL}/pretrained/progress/{session_id}')
        progress_data = response.json()
        
        if progress_data.get('success'):
            print(f"Progress: {progress_data['progress']}% - {progress_data['current_step']} ({progress_data['status']})")
            
            if progress_data['status'] == 'completed':
                print("\nPrediction completed!")
                break
            elif progress_data['status'] == 'error':
                print(f"\nPrediction failed: {progress_data}")
                return False
        
        time.sleep(2)
    else:
        print("\nTimeout waiting for prediction completion")
        return False
    
    # Get results
    print("\n3. Getting results...")
    response = requests.get(f'{BASE_URL}/pretrained/result/{session_id}')
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\nMetrics: {result['metrics']}")
        print(f"Model Info: {result['model_info']}")
        print(f"Number of predictions: {len(result['predictions'])}")
        return True
    else:
        print(f"Error getting results: {result}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Backend API Test Suite")
    print("=" * 60)
    print("\nMake sure the backend server is running on http://localhost:5000")
    print("Run: python backend/app.py")
    input("\nPress Enter to start tests...")
    
    tests = [
        ("Health Check", test_health),
        ("List Datasets", test_datasets),
        ("List Models", test_models),
        ("Custom Training Flow", test_custom_training),
        ("Pretrained Model Flow", test_pretrained)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nError in {name}: {str(e)}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
