import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import { 
  Button, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper,
  Typography,
  Card,
  CardContent,
  Chip,
  LinearProgress
} from '@mui/material';

function SimpleResultPage() {
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadResults = async () => {
      try {
        // Try to get session ID
        const sessionId = localStorage.getItem('trainingSessionId');
        
        if (!sessionId) {
          setError('No training session found. Please train a model first.');
          setLoading(false);
          return;
        }

        // Fetch results from backend
        const response = await fetch(`/api/training/results?session_id=${sessionId}`);
        
        if (!response.ok) {
          throw new Error('Failed to load results');
        }
        
        const data = await response.json();
        setResults(data);
        
        // Clean up session after successful load
        localStorage.removeItem('trainingSessionId');
        
      } catch (err) {
        console.error('Error loading results:', err);
        setError('Failed to load results: ' + err.message);
      } finally {
        setLoading(false);
      }
    };

    loadResults();
  }, []);

  const getChipColor = (predicted) => {
    if (predicted === 'Confirmed Exoplanet') return 'success';
    if (predicted === 'Exoplanet Candidate') return 'warning';
    return 'error';
  };

  if (loading) {
    return (
      <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
        <img className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover" alt="Space background" src="/background.svg" />
        <Navbar />
        <main className="relative z-10 px-8 pt-32 max-w-6xl mx-auto">
          <Typography variant="h4" className="text-white text-center mb-8">
            Loading Results...
          </Typography>
          <LinearProgress />
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
        <img className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover" alt="Space background" src="/background.svg" />
        <Navbar />
        <main className="relative z-10 px-8 pt-32 max-w-6xl mx-auto">
          <Card className="bg-red-900/60 backdrop-blur-sm border border-red-600/30">
            <CardContent className="text-center p-8">
              <Typography variant="h5" className="text-red-200 mb-4">Error</Typography>
              <Typography variant="body1" className="text-red-300 mb-6">{error}</Typography>
              <Button variant="contained" onClick={() => navigate('/select')} sx={{ backgroundColor: '#dc2626' }}>
                Start New Training
              </Button>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover" alt="Space background" src="/background.svg" />
      <Navbar />

      <main className="relative z-10 px-8 pt-32 max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <Typography variant="h3" className="text-white font-bold mb-4">
            Training Results
          </Typography>
          <Typography variant="h6" className="text-gray-300">
            {results?.model_type} • {results?.dataset} Dataset
          </Typography>
        </div>

        {/* Metrics Card */}
        <Card className="bg-gradient-to-br from-blue-900/60 to-blue-800/60 backdrop-blur-sm border border-blue-600/30 shadow-2xl mb-8">
          <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
              <div>
                <Typography variant="h4" className="text-blue-200 font-bold">
                  {(results?.accuracy * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body1" className="text-gray-300">Accuracy</Typography>
              </div>
              <div>
                <Typography variant="h4" className="text-green-200 font-bold">
                  {results?.total_predictions}
                </Typography>
                <Typography variant="body1" className="text-gray-300">Predictions</Typography>
              </div>
              <div>
                <Typography variant="h4" className="text-purple-200 font-bold">
                  {results?.predictions?.filter(p => p.correct).length || 0}
                </Typography>
                <Typography variant="body1" className="text-gray-300">Correct</Typography>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Predictions Table */}
        <Card className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 shadow-2xl mb-8">
          <CardContent className="p-6">
            <Typography variant="h5" className="text-white font-semibold mb-6 text-center">
              Prediction Results (First 20 Objects)
            </Typography>
            
            <TableContainer component={Paper} className="bg-gray-800/40 backdrop-blur-sm">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell className="text-gray-300 font-semibold">Object</TableCell>
                    <TableCell className="text-gray-300 font-semibold">Predicted Type</TableCell>
                    <TableCell className="text-gray-300 font-semibold">Confidence</TableCell>
                    <TableCell className="text-gray-300 font-semibold">Actual</TableCell>
                    <TableCell className="text-gray-300 font-semibold">Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {results?.predictions?.map((pred) => (
                    <TableRow key={pred.id} className="hover:bg-gray-700/30">
                      <TableCell className="text-gray-300">{pred.name}</TableCell>
                      <TableCell>
                        <Chip 
                          label={pred.predicted} 
                          size="small"
                          color={getChipColor(pred.predicted)}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell className="text-gray-300">{(pred.confidence * 100).toFixed(1)}%</TableCell>
                      <TableCell className="text-gray-300">{pred.actual}</TableCell>
                      <TableCell>
                        <Chip 
                          label={pred.correct ? "Correct" : "Incorrect"} 
                          size="small"
                          color={pred.correct ? "success" : "error"}
                          variant="filled"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>

        {/* Action Buttons */}
        <div className="text-center space-x-4 mb-8">
          <Button 
            variant="contained" 
            size="large"
            onClick={() => navigate('/select')}
            sx={{ backgroundColor: '#2563eb', '&:hover': { backgroundColor: '#1d4ed8' } }}
          >
            Train New Model
          </Button>
          <Button 
            variant="outlined" 
            size="large"
            onClick={() => navigate('/predict')}
            sx={{ borderColor: '#10b981', color: '#10b981', '&:hover': { borderColor: '#059669', backgroundColor: '#059669/10' } }}
          >
            Make Predictions
          </Button>
        </div>
      </main>
    </div>
  );
}

export default SimpleResultPage;