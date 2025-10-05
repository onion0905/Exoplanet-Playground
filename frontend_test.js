console.log('=== TESTING FRONTEND WITH SIMPLE SESSION ===');

// Set the working session ID
const sessionId = 'session_4c23ec82_1759637182';
console.log('Setting session ID:', sessionId);

// Store session ID for frontend to pick up
localStorage.setItem('trainingSessionId', sessionId);

console.log('Session ID stored. Reloading page...');
console.log('Check that:');
console.log('1. Loading spinner appears');
console.log('2. Results page loads with metrics');
console.log('3. Prediction table shows 20 objects');
console.log('4. Accuracy and stats are displayed');

// Reload page to trigger results loading
location.reload();