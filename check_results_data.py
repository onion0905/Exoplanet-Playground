import requests
import json

# Check what data the results endpoint actually returns
base_url = 'http://localhost:5000'

try:
    # Start a quick training to get a session
    training_data = {
        'dataset_name': 'kepler',
        'dataset_source': 'nasa', 
        'target_column': 'koi_disposition',
        'model_type': 'random_forest',
        'hyperparameters': {'n_estimators': 3, 'max_depth': 2}
    }
    
    response = requests.post(f'{base_url}/api/training/start', json=training_data)
    if response.status_code == 200:
        session_id = response.json()['session_id']
        print(f'Session: {session_id}')
        
        # Wait for completion
        import time
        for _ in range(30):
            prog_resp = requests.get(f'{base_url}/api/training/progress/{session_id}')
            if prog_resp.status_code == 200:
                prog = prog_resp.json()
                if prog['status'] == 'completed':
                    # Get results and examine structure
                    results_resp = requests.get(f'{base_url}/api/training/results/{session_id}')
                    if results_resp.status_code == 200:
                        results = results_resp.json()
                        print('Results structure:')
                        
                        # Print top-level keys
                        print(f'Top-level keys: {list(results.keys())}')
                        
                        # Check specific keys
                        if 'testing_results' in results:
                            testing = results['testing_results']
                            print(f'Testing results keys: {list(testing.keys())}')
                            
                            if 'confusion_matrix' in testing:
                                cm = testing['confusion_matrix']
                                print(f'Confusion matrix keys: {list(cm.keys())}')
                                if 'matrix' in cm:
                                    matrix = cm['matrix']
                                    print(f'Matrix shape: {len(matrix)} x {len(matrix[0]) if matrix else 0}')
                                print(f'Class names: {cm.get("labels", [])}')
                            
                            if 'summary_metrics' in testing:
                                metrics = testing['summary_metrics']
                                print(f'Summary metrics: {metrics}')
                            
                            if 'predictions_table' in testing:
                                pred_table = testing['predictions_table']
                                print(f'Predictions table length: {len(pred_table)}')
                                if pred_table:
                                    print(f'First prediction sample keys: {list(pred_table[0].keys())}')
                        
                        break
                    else:
                        print(f'Results error: {results_resp.status_code}')
                        break
                elif prog['status'] in ['failed', 'error']:
                    print(f'Training failed: {prog.get("error")}')
                    break
            time.sleep(1)
    else:
        print(f'Training start failed: {response.status_code}')
        
except Exception as e:
    print(f'Error: {e}')