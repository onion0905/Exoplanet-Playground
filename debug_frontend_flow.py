import requests
import json

# Simulate the exact training page flow
base_url = 'http://localhost:5000'
test_session = 'session_858c2e81_1759636121'

print('Testing training page data flow...')

try:
    # Get results like training page does
    results_response = requests.get(f'{base_url}/api/training/results?session_id={test_session}')
    
    if results_response.status_code == 200:
        results_data = results_response.json()
        
        print('\n=== TRAINING PAGE STORES THIS DATA ===')
        print(f'Top-level keys: {list(results_data.keys())}')
        
        if 'testing_results' in results_data:
            testing = results_data['testing_results']
            print(f'testing_results keys: {list(testing.keys())}')
            
            if 'predictions_table' in testing:
                print(f'predictions_table length: {len(testing["predictions_table"])}')
                
                if testing["predictions_table"]:
                    sample = testing["predictions_table"][0]
                    print(f'Sample prediction keys: {list(sample.keys())}')
                    print(f'Sample predicted_label: {sample.get("predicted_label")}')
                    print(f'Sample confidence: {sample.get("confidence")}')
        
        print('\n=== FRONTEND SHOULD PARSE THIS AS ===')
        
        # Simulate frontend parsing logic
        testData = None
        
        # Check for testing_results.predictions_table (new format)
        if results_data.get('testing_results', {}).get('predictions_table') and isinstance(results_data['testing_results']['predictions_table'], list):
            testData = results_data['testing_results']['predictions_table']
            print('✅ Found predictions_table in testing_results')
        else:
            print('❌ No predictions_table found')
        
        if testData:
            formatted_count = 0
            for item in testData[:5]:  # First 5 items
                predLabel = item.get('predicted_label') or item.get('prediction')
                print(f'   Item {formatted_count + 1}: predicted_label="{predLabel}"')
                formatted_count += 1
            
            print(f'✅ Would format {len(testData)} prediction results')
        else:
            print('❌ No test data to format - this explains "No Results Available"')
        
    else:
        print(f'❌ API error: {results_response.status_code}')
        print(results_response.text)

except Exception as e:
    print(f'❌ Error: {e}')