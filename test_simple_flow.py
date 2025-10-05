import requests
import time

def test_simple_flow():
    """Test the simplified training -> results flow"""
    
    print("=== TESTING SIMPLIFIED FLOW ===")
    
    base_url = 'http://localhost:5000'
    
    try:
        # 1. Start training
        training_data = {
            'dataset_name': 'kepler',
            'dataset_source': 'nasa',  
            'target_column': 'koi_disposition',
            'model_type': 'random_forest',
            'hyperparameters': {'n_estimators': 5, 'max_depth': 3}
        }
        
        print("1. Starting training...")
        response = requests.post(f'{base_url}/api/training/start', json=training_data)
        
        if response.status_code != 200:
            print(f"❌ Training failed: {response.status_code}")
            return
            
        result = response.json()
        session_id = result['session_id']
        print(f"✅ Training started: {session_id}")
        
        # 2. Wait for completion
        print("2. Waiting for completion...")
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
        
        # 3. Test new simple results endpoint
        print("3. Testing simplified results endpoint...")
        results_response = requests.get(f'{base_url}/api/training/results?session_id={session_id}')
        
        if results_response.status_code != 200:
            print(f"❌ Results failed: {results_response.status_code}")
            print(results_response.text)
            return
            
        data = results_response.json()
        print("✅ Results received!")
        
        # 4. Check simple response structure
        print(f"\n4. SIMPLE RESPONSE STRUCTURE:")
        print(f"   Session ID: {data.get('session_id')}")
        print(f"   Accuracy: {data.get('accuracy')}")
        print(f"   Total Predictions: {data.get('total_predictions')}")
        print(f"   Sample Predictions: {len(data.get('predictions', []))}")
        print(f"   Model Type: {data.get('model_type')}")
        print(f"   Dataset: {data.get('dataset')}")
        
        # 5. Show sample predictions
        predictions = data.get('predictions', [])
        if predictions:
            print(f"\n5. SAMPLE PREDICTIONS:")
            for pred in predictions[:5]:
                status = "✅" if pred['correct'] else "❌"
                print(f"   {status} {pred['name']}: {pred['predicted']} ({pred['confidence']*100:.1f}%)")
        
        print(f"\n✅ SIMPLE FLOW WORKING!")
        print(f"Frontend can now fetch: /api/training/results?session_id={session_id}")
        print(f"And display {len(predictions)} predictions in a simple table")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_flow()