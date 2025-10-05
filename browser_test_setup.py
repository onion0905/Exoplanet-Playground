import requests
import json

def create_test_data_for_browser():
    """Create test data and provide instructions for manual browser testing"""
    
    print("=== CREATING TEST DATA FOR BROWSER TESTING ===")
    
    base_url = 'http://localhost:5000'
    
    # Get a working session with results
    session_id = 'session_c12bde24_1759636622'
    
    try:
        # Get the complete results data
        print("1. Fetching complete results data...")
        response = requests.get(f'{base_url}/api/training/results?session_id={session_id}')
        
        if response.status_code != 200:
            print(f"❌ Failed to get results: {response.status_code}")
            return
        
        results_data = response.json()
        print("✅ Results data fetched")
        
        # Create the JavaScript code for browser testing
        results_json = json.dumps(results_data)
        escaped_json = results_json.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        
        print("\n2. BROWSER TESTING INSTRUCTIONS:")
        print("="*50)
        print("Step 1: Open http://localhost:3000/custom_result in your browser")
        print("Step 2: Open browser console (F12)")
        print("Step 3: Copy and paste the following JavaScript code:")
        print("")
        
        # Method 1: Direct localStorage injection (simulating training page)
        print("// METHOD 1: Simulate training page storing results")
        print("console.log('Setting training results in localStorage...');")
        print(f'localStorage.setItem("trainingResults", "{escaped_json}");')
        print("console.log('Results stored, reloading page...');")
        print("location.reload();")
        print("")
        
        # Method 2: Session ID method (simulating fallback)
        print("// METHOD 2: Alternative - use session ID for backend fetch")
        print("console.log('Setting session ID in localStorage...');")
        print(f'localStorage.setItem("trainingSessionId", "{session_id}");')
        print("console.log('Session ID stored, reloading page...');")
        print("location.reload();")
        
        print("")
        print("="*50)
        print("What to expect:")
        print("1. Page should show training results with confusion matrix")
        print("2. Should display 100 prediction results in a table")
        print("3. Should show accuracy metrics")
        print("4. Console should show debug messages about data processing")
        print("")
        print("If you see 'No Results Available':")
        print("1. Check console for error messages")
        print("2. Verify the localStorage data was set correctly")
        print("3. Check network tab for any failed API calls")
        
        # Summary of expected data structure
        print(f"\n3. DATA STRUCTURE SUMMARY:")
        print(f"   Session ID: {session_id}")
        print(f"   Predictions: {len(results_data.get('testing_results', {}).get('predictions_table', []))}")
        print(f"   Accuracy: {results_data.get('testing_results', {}).get('summary_metrics', {}).get('accuracy', 'N/A')}")
        
        # Create a simple verification script
        print(f"\n4. VERIFICATION SCRIPT (run in console after loading data):")
        print("""
// Check if data is loaded correctly
const results = JSON.parse(localStorage.getItem('trainingResults') || '{}');
console.log('Results loaded:', !!results.testing_results);
console.log('Predictions table:', results.testing_results?.predictions_table?.length || 0);
console.log('First prediction:', results.testing_results?.predictions_table?.[0]);

// Check if frontend processed the data
console.log('Page content includes No Results:', document.body.innerHTML.includes('No Results Available'));
console.log('Page has prediction table:', document.querySelector('table') !== null);
""")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_test_data_for_browser()