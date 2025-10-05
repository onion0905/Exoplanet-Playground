import requests
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def test_frontend_results_display():
    print("=== TESTING FRONTEND RESULTS DISPLAY ===")
    
    # First, create a training session with results
    base_url = 'http://localhost:5000'
    
    try:
        # Start training
        training_data = {
            'dataset_name': 'kepler',
            'dataset_source': 'nasa',
            'target_column': 'koi_disposition',
            'model_type': 'random_forest',
            'hyperparameters': {'n_estimators': 5, 'max_depth': 3}
        }
        
        print("1. Starting training session...")
        response = requests.post(f'{base_url}/api/training/start', json=training_data)
        if response.status_code != 200:
            print(f"❌ Training failed to start: {response.status_code}")
            return
            
        result = response.json()
        session_id = result['session_id']
        print(f"✅ Training started: {session_id}")
        
        # Wait for completion
        print("2. Waiting for training completion...")
        for i in range(30):
            progress_response = requests.get(f'{base_url}/api/training/progress?session_id={session_id}')
            if progress_response.status_code == 200:
                progress = progress_response.json()
                
                if progress['status'] == 'completed' or progress.get('ready_for_results'):
                    print("✅ Training completed!")
                    break
                elif progress['status'] in ['failed', 'error']:
                    print(f"❌ Training failed: {progress.get('error')}")
                    return
            time.sleep(2)
        else:
            print("❌ Training timeout")
            return
        
        # Get results data
        print("3. Getting training results...")
        results_response = requests.get(f'{base_url}/api/training/results?session_id={session_id}')
        if results_response.status_code != 200:
            print(f"❌ Failed to get results: {results_response.status_code}")
            return
            
        results_data = results_response.json()
        print(f"✅ Results retrieved - has testing_results: {bool(results_data.get('testing_results'))}")
        print(f"   Predictions table length: {len(results_data.get('testing_results', {}).get('predictions_table', []))}")
        
        # Now test the frontend
        print("\n4. Testing frontend with browser automation...")
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Start browser
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Navigate to results page
            driver.get('http://localhost:3000/custom_result')
            
            # Inject the training results into localStorage (simulating training page flow)
            print("5. Injecting results into localStorage...")
            results_json = json.dumps(results_data)
            
            # Execute JavaScript to set localStorage
            driver.execute_script(f"""
                console.log('Setting trainingResults in localStorage...');
                localStorage.setItem('trainingResults', arguments[0]);
                console.log('localStorage set, reloading page...');
                location.reload();
            """, results_json)
            
            # Wait for page to reload and process
            time.sleep(3)
            
            # Check for console logs
            logs = driver.get_log('browser')
            print("\n6. Console output:")
            for log in logs:
                if log['level'] in ['INFO', 'LOG']:
                    print(f"   {log['message']}")
            
            # Check if "No Results Available" is shown
            try:
                no_results = driver.find_element(By.XPATH, "//*[contains(text(), 'No Results Available')]")
                print("❌ Found 'No Results Available' message - frontend not working correctly")
                
                # Get page source for debugging
                page_source = driver.page_source
                if 'No Results Available' in page_source:
                    print("   Confirmed: No Results Available is displayed")
                    
            except:
                print("✅ No 'No Results Available' message found")
                
                # Check if results table is present
                try:
                    results_table = driver.find_element(By.TAG_NAME, "table")
                    rows = driver.find_elements(By.XPATH, "//table//tr")
                    print(f"✅ Results table found with {len(rows)} rows")
                    
                    # Check for prediction results
                    predictions = driver.find_elements(By.XPATH, "//*[contains(text(), 'Confirmed') or contains(text(), 'False Positive') or contains(text(), 'Candidate')]")
                    print(f"✅ Found {len(predictions)} prediction elements")
                    
                except:
                    print("❌ No results table found")
            
            # Check accuracy display
            try:
                accuracy_element = driver.find_element(By.XPATH, "//*[contains(text(), 'Accuracy')]")
                print("✅ Accuracy display found")
            except:
                print("❌ Accuracy display not found")
            
            print(f"\n7. Session ID for manual testing: {session_id}")
            print("   You can manually navigate to: http://localhost:3000/custom_result")
            print("   And check browser console for debug messages")
            
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frontend_results_display()