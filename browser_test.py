import requests
import json

# Get the latest session data to test in browser
base_url = 'http://localhost:5000'
test_session = 'session_6be21cfb_1759636416'

print('Getting test data for browser localStorage...')

try:
    response = requests.get(f'{base_url}/api/training/results?session_id={test_session}')
    
    if response.status_code == 200:
        results_data = response.json()
        
        # Create JavaScript code to paste in browser console
        json_string = json.dumps(results_data).replace('`', '\\`').replace('\\', '\\\\')
        
        print('\n=== BROWSER CONSOLE TEST CODE ===')
        print('Copy and paste this into browser console on results page:')
        print('')
        print('// Clear any existing data')
        print('localStorage.clear();')
        print('')
        print('// Set test results data')
        print(f'localStorage.setItem("trainingResults", `{json_string}`);')
        print('')
        print('// Refresh page to see results')
        print('location.reload();')
        print('')
        
        print('\n=== VERIFICATION ===')
        print('After running the above code, you should see:')
        print(f'- {len(results_data["testing_results"]["predictions_table"])} prediction results')
        
        # Count predictions by type for verification
        pred_table = results_data["testing_results"]["predictions_table"]
        pred_counts = {}
        for item in pred_table:
            pred_label = item.get('predicted_label', 'unknown')
            pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1
        
        print(f'- Prediction distribution: {pred_counts}')
        print('- Confusion matrix with 3x3 grid')
        print('- Summary metrics (accuracy, precision, recall, F1)')
        
    else:
        print(f'❌ Error getting test data: {response.status_code}')
        
except Exception as e:
    print(f'❌ Error: {e}')