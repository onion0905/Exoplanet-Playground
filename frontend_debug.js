// Test script to check frontend session and localStorage
console.log('=== FRONTEND SESSION TEST ===');

// Check localStorage
console.log('Training session ID:', localStorage.getItem('trainingSessionId'));
console.log('Training results:', localStorage.getItem('trainingResults'));

// Check URL parameters
const urlParams = new URLSearchParams(window.location.search);
console.log('URL session_id parameter:', urlParams.get('session_id'));

// Try to fetch results with known session ID
const testSessionId = 'session_858c2e81_1759636121'; // From our test

fetch(`/api/training/results?session_id=${testSessionId}`)
  .then(response => response.json())
  .then(data => {
    console.log('Direct API test - Response structure:', Object.keys(data));
    console.log('Has testing_results:', !!data.testing_results);
    console.log('Has predictions_table:', !!data.testing_results?.predictions_table);
    console.log('Predictions count:', data.testing_results?.predictions_table?.length);
  })
  .catch(error => console.error('API test error:', error));