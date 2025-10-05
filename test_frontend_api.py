import requests

def test_frontend_api_access():
    """Test if frontend can access backend APIs correctly"""
    
    print("=== TESTING FRONTEND API ACCESS ===")
    
    # Test with a known working session
    session_id = 'session_c12bde24_1759636622'
    
    try:
        # Test the route that frontend uses
        frontend_api_url = 'http://localhost:5000/api/training/results'
        
        # Test with query parameter (how frontend calls it)
        print("1. Testing frontend API call with query parameter...")
        response = requests.get(f'{frontend_api_url}?session_id={session_id}')
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Frontend API call successful")
            print(f"   Data keys: {list(data.keys())}")
            print(f"   Has testing_results: {bool(data.get('testing_results'))}")
            print(f"   Predictions count: {len(data.get('testing_results', {}).get('predictions_table', []))}")
        else:
            print(f"   ❌ API call failed: {response.text}")
        
        # Test CORS headers
        print("\n2. Testing CORS headers...")
        headers = response.headers
        print(f"   Access-Control-Allow-Origin: {headers.get('Access-Control-Allow-Origin', 'NOT SET')}")
        print(f"   Access-Control-Allow-Methods: {headers.get('Access-Control-Allow-Methods', 'NOT SET')}")
        
        # Test from frontend origin
        print("\n3. Testing with frontend origin...")
        test_headers = {'Origin': 'http://localhost:3000'}
        response_with_origin = requests.get(f'{frontend_api_url}?session_id={session_id}', headers=test_headers)
        print(f"   Status with origin: {response_with_origin.status_code}")
        
        # Test different session formats
        print("\n4. Testing session ID formats...")
        for test_session in [session_id, session_id.replace('session_', ''), 'invalid_session']:
            test_resp = requests.get(f'{frontend_api_url}?session_id={test_session}')
            print(f"   Session '{test_session}': {test_resp.status_code}")
        
        print(f"\n5. ✅ Session for frontend testing: {session_id}")
        print("   Open http://localhost:3000/custom_result in browser")
        print("   Open console and run:")
        print(f"   localStorage.setItem('trainingSessionId', '{session_id}');")
        print("   location.reload();")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_frontend_api_access()