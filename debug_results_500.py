import requests
import json

# Debug the 500 error in results endpoint
base_url = 'http://localhost:5000'

print('Debugging results 500 error...')

# First, let's start a training to get a fresh session
training_data = {
    'dataset_name': 'kepler',
    'dataset_source': 'nasa',
    'target_column': 'koi_disposition',
    'model_type': 'random_forest',
    'hyperparameters': {'n_estimators': 5, 'max_depth': 3}
}

try:
    print('1. Starting training session...')
    response = requests.post(f'{base_url}/api/training/start', json=training_data)
    
    if response.status_code == 200:
        result = response.json()
        session_id = result['session_id']
        print(f'   Session ID: {session_id}')
        
        # Wait for completion
        import time
        for i in range(30):
            progress_response = requests.get(f'{base_url}/api/training/progress/{session_id}')
            if progress_response.status_code == 200:
                progress = progress_response.json()
                print(f'   Status: {progress["status"]} ({progress.get("progress", 0)}%)')
                
                if progress['status'] == 'completed':
                    print('2. Training completed! Now testing results endpoint...')
                    
                    # Test results endpoint with detailed error info
                    try:
                        results_response = requests.get(f'{base_url}/api/training/results/{session_id}')
                        print(f'   Results status: {results_response.status_code}')
                        
                        if results_response.status_code == 200:
                            results = results_response.json()
                            print('✅ Results fetched successfully!')
                            print(f'   Keys in results: {list(results.keys())}')
                        else:
                            print(f'❌ Results error: {results_response.status_code}')
                            try:
                                error_info = results_response.json()
                                print(f'   Error details: {error_info}')
                            except:
                                print(f'   Raw error response: {results_response.text}')
                    
                    except Exception as e:
                        print(f'❌ Exception fetching results: {e}')
                    
                    break
                elif progress['status'] in ['failed', 'error']:
                    print(f'❌ Training failed: {progress.get("error", "Unknown error")}')
                    break
            
            time.sleep(1)
        else:
            print('⏰ Training timeout')
    else:
        print(f'❌ Training start failed: {response.status_code}')
        print(f'   Error: {response.text}')

except requests.exceptions.ConnectionError:
    print('❌ Could not connect to backend. Please ensure Flask server is running on localhost:5000')
except Exception as e:
    print(f'❌ Unexpected error: {e}')