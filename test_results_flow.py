import requests
import json
import time

def test_results_page_data():
    print("=== TESTING RESULTS PAGE DATA FLOW ===")
    
    base_url = 'http://localhost:5000'
    frontend_url = 'http://localhost:3000'
    
    try:
        # 1. Create a training session
        training_data = {
            'dataset_name': 'kepler',
            'dataset_source': 'nasa',
            'target_column': 'koi_disposition',
            'model_type': 'random_forest',
            'hyperparameters': {'n_estimators': 5, 'max_depth': 3}
        }
        
        print("1. Starting new training session...")
        response = requests.post(f'{base_url}/api/training/start', json=training_data)
        if response.status_code != 200:
            print(f"❌ Training failed: {response.status_code}")
            return
            
        result = response.json()
        session_id = result['session_id']
        print(f"✅ Training started: {session_id}")
        
        # 2. Wait for completion
        print("2. Waiting for training completion...")
        for i in range(30):
            progress_response = requests.get(f'{base_url}/api/training/progress?session_id={session_id}')
            if progress_response.status_code == 200:
                progress = progress_response.json()
                
                if progress['status'] == 'completed':
                    print("✅ Training completed!")
                    break
                elif progress['status'] in ['failed', 'error']:
                    print(f"❌ Training failed: {progress.get('error')}")
                    return
            time.sleep(2)
        else:
            print("❌ Training timeout")
            return
        
        # 3. Get results (like training page does)
        print("3. Fetching training results...")
        results_response = requests.get(f'{base_url}/api/training/results?session_id={session_id}')
        if results_response.status_code != 200:
            print(f"❌ Results fetch failed: {results_response.status_code}")
            return
            
        results_data = results_response.json()
        print("✅ Results fetched successfully")
        
        # 4. Analyze data structure (what training page stores)
        print("\n4. ANALYZING DATA STRUCTURE FOR FRONTEND...")
        print(f"   Top-level keys: {list(results_data.keys())}")
        
        if 'testing_results' in results_data:
            testing = results_data['testing_results']
            print(f"   testing_results keys: {list(testing.keys())}")
            
            if 'predictions_table' in testing:
                pred_table = testing['predictions_table']
                print(f"   predictions_table length: {len(pred_table)}")
                print(f"   predictions_table type: {type(pred_table)}")
                print(f"   Is array: {isinstance(pred_table, list)}")
                
                if pred_table:
                    sample = pred_table[0]
                    print(f"   Sample keys: {list(sample.keys())}")
                    print(f"   Sample predicted_label: '{sample.get('predicted_label')}'")
                    print(f"   Sample confidence: {sample.get('confidence')}")
            else:
                print("   ❌ No predictions_table in testing_results")
        else:
            print("   ❌ No testing_results in response")
        
        # 5. Test frontend condition exactly
        print("\n5. TESTING FRONTEND PARSING CONDITION...")
        
        # Exact condition from CustomResultPage.jsx
        if (results_data.get('testing_results', {}).get('predictions_table') and 
            isinstance(results_data['testing_results']['predictions_table'], list)):
            
            test_data = results_data['testing_results']['predictions_table']
            print(f"   ✅ Frontend condition PASSED")
            print(f"   Would process {len(test_data)} items")
            
            # Test first few items processing
            for i, item in enumerate(test_data[:3]):
                pred_label = item.get('predicted_label') or item.get('prediction')
                confidence = item.get('confidence', 0.5)
                
                if pred_label == 'planet':
                    prediction = "Confirmed Exoplanet"
                elif pred_label == 'candidate':
                    prediction = "Exoplanet Candidate"
                elif pred_label == 'false_positive':
                    prediction = "False Positive"
                else:
                    prediction = str(pred_label)
                
                print(f"      Item {i+1}: '{pred_label}' -> '{prediction}' ({confidence:.3f})")
            
            print(f"   ✅ Frontend would show {len(test_data)} prediction results")
            
        else:
            print("   ❌ Frontend condition FAILED")
            print(f"      testing_results exists: {bool(results_data.get('testing_results'))}")
            if results_data.get('testing_results'):
                print(f"      predictions_table exists: {bool(results_data['testing_results'].get('predictions_table'))}")
                pt = results_data['testing_results'].get('predictions_table')
                print(f"      predictions_table type: {type(pt)}")
                print(f"      predictions_table is list: {isinstance(pt, list)}")
        
        # 6. Generate localStorage data for manual testing
        print(f"\n6. MANUAL TESTING INSTRUCTIONS:")
        print(f"   Session ID: {session_id}")
        print(f"   Frontend URL: {frontend_url}/custom_result")
        print(f"   \n   To manually test:")
        print(f"   1. Open {frontend_url}/custom_result in browser")
        print(f"   2. Open browser console (F12)")
        print(f"   3. Run this JavaScript:")
        
        # Create the exact localStorage data that training page would store
        storage_data = json.dumps(results_data).replace('"', '\\"')
        print(f'''
   localStorage.setItem("trainingResults", "{storage_data[:200]}...");
   location.reload();
   ''')
        
        print(f"   4. Check console for debug messages from CustomResultPage.jsx")
        print(f"   5. Verify if 'No Results Available' is shown or if results appear")
        
        # 7. Alternative: Test backend route directly from frontend
        print(f"\n7. ALTERNATIVE - Test backend route from frontend:")
        print(f"   1. Set session ID: localStorage.setItem('trainingSessionId', '{session_id}');")
        print(f"   2. Refresh page - frontend should fetch from /api/training/results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_results_page_data()