import requests
import json
import time

# Test backend training endpoint
base_url = 'http://localhost:5000'

print('Testing backend three-class training...')

# Start training
training_data = {
    'dataset_name': 'kepler',
    'dataset_source': 'nasa',
    'target_column': 'koi_disposition',
    'model_type': 'random_forest',
    'hyperparameters': {'n_estimators': 10, 'max_depth': 3}
}

print('1. Starting training session...')
try:
    response = requests.post(f'{base_url}/api/training/start', json=training_data)
    print(f'Status: {response.status_code}')

    if response.status_code == 200:
        result = response.json()
        session_id = result['session_id']
        print(f'Session ID: {session_id}')
        
        # Check progress
        max_checks = 20
        for i in range(max_checks):
            print(f'2. Checking progress ({i+1}/{max_checks})...')
            progress_response = requests.get(f'{base_url}/api/training/progress/{session_id}')
            
            if progress_response.status_code == 200:
                progress = progress_response.json()
                print(f'   Status: {progress["status"]}')
                
                if progress['status'] == 'completed':
                    print('3. Training completed! Checking results...')
                    
                    # Get results
                    results_response = requests.get(f'{base_url}/api/training/results/{session_id}')
                    if results_response.status_code == 200:
                        results = results_response.json()
                        
                        # Check for three-class structure
                        if 'results' in results and 'confusion_matrix' in results['results']:
                            cm = results['results']['confusion_matrix']
                            print(f'   Confusion matrix shape: {len(cm)} x {len(cm[0]) if cm else 0}')
                            
                            if 'class_names' in results['results']:
                                print(f'   Classes: {results["results"]["class_names"]}')
                            
                            if 'classification_report' in results['results']:
                                print('   Classification report found')
                        
                        print('✅ Three-class training completed successfully!')
                        break
                    else:
                        print(f'   Results request failed: {results_response.status_code}')
                elif progress['status'] == 'failed' or progress['status'] == 'error':
                    print(f'❌ Training failed: {progress.get("error", progress.get("message", "Unknown error"))}')
                    break
            else:
                print(f'   Progress request failed: {progress_response.status_code}')
            
            time.sleep(2)
        else:
            print('⏰ Training timeout')
    else:
        print(f'❌ Training start failed: {response.status_code}')
        if response.headers.get('content-type', '').startswith('application/json'):
            print(f'Error: {response.json()}')
        else:
            print(f'Error: {response.text}')

except requests.exceptions.ConnectionError:
    print('❌ Could not connect to backend server. Make sure Flask server is running on localhost:5000')
except Exception as e:
    print(f'❌ Error: {e}')