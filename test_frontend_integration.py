import requests
import json
import time

# Test end-to-end three-class system with frontend data
base_url = 'http://localhost:5000'

print('Testing complete three-class frontend integration...')

try:
    # Start training
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
        
        # Wait for completion
        for i in range(30):
            progress_response = requests.get(f'{base_url}/api/training/progress/{session_id}')
            if progress_response.status_code == 200:
                progress = progress_response.json()
                print(f'   Progress: {progress["status"]} ({progress.get("progress", 0)}%)')
                
                if progress['status'] == 'completed':
                    print('✅ Training completed!')
                    
                    # Get results
                    results_response = requests.get(f'{base_url}/api/training/results/{session_id}')
                    if results_response.status_code == 200:
                        results = results_response.json()
                        
                        print('\n=== FRONTEND DATA ANALYSIS ===')
                        
                        # Check predictions table for frontend
                        if 'testing_results' in results and 'predictions_table' in results['testing_results']:
                            pred_table = results['testing_results']['predictions_table']
                            print(f'✅ Predictions table: {len(pred_table)} entries')
                            
                            if pred_table:
                                sample = pred_table[0]
                                print(f'   Sample prediction keys: {list(sample.keys())}')
                                print(f'   Sample predicted_label: {sample.get("predicted_label")}')
                                print(f'   Sample confidence: {sample.get("confidence")}')
                                print(f'   Sample true_label: {sample.get("true_label")}')
                                
                                # Count predictions by type
                                pred_counts = {}
                                for item in pred_table:
                                    pred_label = item.get('predicted_label', 'unknown')
                                    pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1
                                
                                print(f'   Prediction distribution: {pred_counts}')
                        
                        # Check confusion matrix for frontend
                        if 'testing_results' in results and 'confusion_matrix' in results['testing_results']:
                            cm = results['testing_results']['confusion_matrix']
                            print(f'✅ Confusion matrix available')
                            print(f'   Matrix shape: {len(cm["matrix"])} x {len(cm["matrix"][0])}')
                            print(f'   Class labels: {cm["labels"]}')
                        
                        # Check summary metrics for frontend
                        if 'testing_results' in results and 'summary_metrics' in results['testing_results']:
                            metrics = results['testing_results']['summary_metrics']
                            print(f'✅ Summary metrics available')
                            print(f'   Accuracy: {metrics.get("accuracy", 0):.3f}')
                            print(f'   Precision: {metrics.get("precision", 0):.3f}')
                            print(f'   Recall: {metrics.get("recall", 0):.3f}')
                            print(f'   F1-Score: {metrics.get("f1_score", 0):.3f}')
                        
                        print(f'\n🎯 Session ID for frontend testing: {session_id}')
                        print('   You can now navigate to /result page with this session!')
                        
                    else:
                        print(f'❌ Results error: {results_response.status_code}')
                    
                    break
                elif progress['status'] in ['failed', 'error']:
                    print(f'❌ Training failed: {progress.get("error")}')
                    break
            
            time.sleep(2)
        else:
            print('⏰ Training timeout')
    else:
        print(f'❌ Training start failed: {response.status_code}')

except Exception as e:
    print(f'❌ Error: {e}')