import requests
import json

# Test complete flow including localStorage simulation
base_url = 'http://localhost:5000'

print('Testing complete training -> results flow...')

try:
    # Start new training
    training_data = {
        'dataset_name': 'kepler',
        'dataset_source': 'nasa',
        'target_column': 'koi_disposition',
        'model_type': 'random_forest',
        'hyperparameters': {'n_estimators': 5, 'max_depth': 3}
    }
    
    response = requests.post(f'{base_url}/api/training/start', json=training_data)
    if response.status_code == 200:
        result = response.json()
        session_id = result['session_id']
        print(f'✅ Training started: {session_id}')
        
        # Poll for completion (like training page does)
        import time
        for i in range(30):
            progress_response = requests.get(f'{base_url}/api/training/progress?session_id={session_id}')
            if progress_response.status_code == 200:
                progress = progress_response.json()
                
                if progress['status'] == 'completed' or progress.get('ready_for_results'):
                    print(f'✅ Training completed!')
                    
                    # Get results (like training page does)
                    results_response = requests.get(f'{base_url}/api/training/results?session_id={session_id}')
                    if results_response.status_code == 200:
                        results_data = results_response.json()
                        
                        print('\n=== SIMULATING LOCALSTORAGE STORE/RETRIEVE ===')
                        # Simulate localStorage.setItem('trainingResults', JSON.stringify(resultsData))
                        stored_json = json.dumps(results_data)
                        
                        # Simulate localStorage.getItem('trainingResults') and JSON.parse()
                        parsed_results = json.loads(stored_json)
                        
                        print(f'Stored and parsed successfully')
                        print(f'Top-level keys: {list(parsed_results.keys())}')
                        print(f'Has testing_results: {bool(parsed_results.get("testing_results"))}')
                        
                        if parsed_results.get('testing_results'):
                            testing = parsed_results['testing_results']
                            print(f'testing_results keys: {list(testing.keys())}')
                            print(f'Has predictions_table: {bool(testing.get("predictions_table"))}')
                            print(f'predictions_table is array: {isinstance(testing.get("predictions_table"), list)}')
                            
                            if testing.get('predictions_table'):
                                print(f'predictions_table length: {len(testing["predictions_table"])}')
                        
                        # Test frontend condition exactly
                        test_data = None
                        if (parsed_results.get('testing_results', {}).get('predictions_table') and 
                            isinstance(parsed_results['testing_results']['predictions_table'], list)):
                            test_data = parsed_results['testing_results']['predictions_table']
                            print(f'✅ Frontend condition PASSED - would process {len(test_data)} items')
                        else:
                            print('❌ Frontend condition FAILED')
                            print('  testing_results exists:', bool(parsed_results.get('testing_results')))
                            if parsed_results.get('testing_results'):
                                print('  predictions_table exists:', bool(parsed_results['testing_results'].get('predictions_table')))
                                print('  predictions_table is list:', isinstance(parsed_results['testing_results'].get('predictions_table'), list))
                        
                        print(f'\n🔗 Test session ID: {session_id}')
                        print('You can manually test this by:')
                        print('1. Opening browser console on results page')
                        print(f'2. Run: localStorage.setItem("trainingResults", `{stored_json[:100]}...`)')
                        print('3. Refresh the page')
                        
                    else:
                        print(f'❌ Results fetch failed: {results_response.status_code}')
                    break
                elif progress['status'] in ['failed', 'error']:
                    print(f'❌ Training failed: {progress.get("error")}')
                    break
            
            time.sleep(2)
        
    else:
        print(f'❌ Training start failed: {response.status_code} - {response.text}')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()