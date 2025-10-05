import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import ExoplanetVisualization from "../../components/ExoplanetVisualization";
import StarVisualization from "../../components/StarVisualization";
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
  Box,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Card,
  CardContent,
  Alert,
  CircularProgress
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import AddIcon from '@mui/icons-material/Add';
import DownloadIcon from '@mui/icons-material/Download';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { pretrainedPrediction } from '../../lib/api';

function PretrainedResultPage() {
  const navigate = useNavigate();
  const [expandedRows, setExpandedRows] = useState({});
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedPlanet, setSelectedPlanet] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      const sessionId = sessionStorage.getItem('pretrained_session_id');
      
      if (!sessionId) {
        setError('No prediction session found. Please start prediction first.');
        setTimeout(() => navigate("/pretrained"), 2000);
        return;
      }

      try {
        const data = await pretrainedPrediction.getResult(sessionId);
        
        if (data.success) {
          setMetrics(data.metrics);
          setPredictions(data.predictions || []);
          setModelInfo(data.model_info);
        } else {
          setError(data.error || 'Failed to get results');
        }
      } catch (err) {
        console.error('Results fetch error:', err);
        setError(err.message || 'Failed to load results');
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [navigate]);

  const handleExpandRow = (planetId) => {
    setExpandedRows(prev => ({
      ...prev,
      [planetId]: !prev[planetId]
    }));
  };

  const handleView3D = (planet) => {
    setSelectedPlanet(planet);
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setSelectedPlanet(null);
  };

  const handleDownloadResults = () => {
    if (!predictions || predictions.length === 0) {
      return;
    }
    
    const csvContent = "data:text/csv;charset=utf-8," + 
      "ID,Name,Prediction,Confidence\n" +
      predictions.map(planet => 
        `${planet.id || ''},${planet.name || ''},${planet.prediction},${planet.confidence}`
      ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "exoplanet_predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#10b981'; // green
    if (confidence >= 0.8) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };

  const getPredictionColor = (prediction) => {
    const pred = prediction?.toUpperCase() || '';
    if (pred.includes('EXOPLANET') || pred.includes('PLANET')) return '#10b981'; // green
    if (pred.includes('CANDIDATE')) return '#f59e0b'; // yellow/orange
    return '#ef4444'; // red for false positive
  };

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img
        className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover"
        alt="Space background"
        src="/background.svg"
      />

      <Navbar />

      <main className="relative z-10 px-[3.375rem] pt-32 max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h1 className="font-bold text-white text-5xl mb-6">
            Prediction Results
          </h1>
          <p className="text-white text-xl mb-8">
            Analysis complete! Here are the exoplanet classification results.
          </p>
        </div>

        {loading && (
          <div className="text-center py-20">
            <CircularProgress size={60} />
            <Typography variant="h6" className="text-white mt-4">
              Loading results...
            </Typography>
          </div>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 4 }}>
            {error}
          </Alert>
        )}

        {!loading && !error && metrics && (
          <>
            {/* 結果統計卡片 */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
              <Card className="bg-gradient-to-br from-blue-800/60 to-blue-900/60 backdrop-blur-sm border border-blue-600/30">
                <CardContent className="text-center p-6">
                  <Typography variant="h4" className="text-blue-400 font-bold mb-2">
                    {(metrics.accuracy * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" className="text-white">
                    Accuracy
                  </Typography>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-green-800/60 to-green-900/60 backdrop-blur-sm border border-green-600/30">
                <CardContent className="text-center p-6">
                  <Typography variant="h4" className="text-green-400 font-bold mb-2">
                    {predictions.filter(p => p.prediction === 'Exoplanet').length}
                  </Typography>
                  <Typography variant="body1" className="text-white">
                    Exoplanets Detected
                  </Typography>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-yellow-800/60 to-yellow-900/60 backdrop-blur-sm border border-yellow-600/30">
                <CardContent className="text-center p-6">
                  <Typography variant="h4" className="text-yellow-400 font-bold mb-2">
                    {predictions.filter(p => p.prediction === 'Candidate').length}
                  </Typography>
                  <Typography variant="body1" className="text-white">
                    Candidates
                  </Typography>
                </CardContent>
              </Card>
              
              <Card className="bg-gradient-to-br from-red-800/60 to-red-900/60 backdrop-blur-sm border border-red-600/30">
                <CardContent className="text-center p-6">
                  <Typography variant="h4" className="text-red-400 font-bold mb-2">
                    {predictions.filter(p => p.prediction === 'False Positive').length}
                  </Typography>
                  <Typography variant="body1" className="text-white">
                    False Positives
                  </Typography>
                </CardContent>
              </Card>
              
              <Card className="bg-gradient-to-br from-purple-800/60 to-purple-900/60 backdrop-blur-sm border border-purple-600/30">
                <CardContent className="text-center p-6">
                  <Typography variant="h4" className="text-purple-400 font-bold mb-2">
                    {predictions.length}
                  </Typography>
                  <Typography variant="body1" className="text-white">
                    Total Predictions
                  </Typography>
                </CardContent>
              </Card>
            </div>

            {/* 預測結果表格 */}
            <Card className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 shadow-2xl mb-8">
              <CardContent className="p-0">
                <div className="p-6 border-b border-gray-600/30">
                  <div className="flex items-center justify-between">
                    <Typography variant="h5" className="text-white font-semibold">
                      Detailed Results
                    </Typography>
                    <div className="flex gap-2">
                      <Button
                        variant="outlined"
                        startIcon={<AddIcon />}
                        onClick={() => navigate("/pretrained")}
                        sx={{
                          color: 'white',
                          borderColor: 'rgba(255, 255, 255, 0.3)',
                          '&:hover': {
                            borderColor: 'white',
                            backgroundColor: 'rgba(255, 255, 255, 0.1)'
                          }
                        }}
                      >
                        New Prediction
                      </Button>
                      <Button
                        variant="contained"
                        startIcon={<DownloadIcon />}
                        onClick={handleDownloadResults}
                        sx={{
                          backgroundColor: '#2563eb',
                          '&:hover': {
                            backgroundColor: '#1d4ed8'
                          }
                        }}
                      >
                        Download Results
                      </Button>
                    </div>
                  </div>
                </div>
                
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow className="bg-gray-700/30">
                        <TableCell className="text-white font-semibold">Planet Name</TableCell>
                        <TableCell className="text-white font-semibold">Prediction</TableCell>
                        <TableCell className="text-white font-semibold">Confidence</TableCell>
                        <TableCell className="text-white font-semibold">3D View</TableCell>
                        <TableCell className="text-white font-semibold">Details</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {predictions.map((planet) => (
                        <React.Fragment key={planet.id}>
                          <TableRow className="hover:bg-gray-700/20">
                            <TableCell className="text-white font-medium">
                              {planet.name || `Sample ${planet.id}`}
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={planet.prediction}
                                sx={{
                                  backgroundColor: getPredictionColor(planet.prediction),
                                  color: 'white',
                                  fontWeight: 'bold'
                                }}
                              />
                            </TableCell>
                            <TableCell>
                              <Box className="flex items-center gap-2">
                                <LinearProgress
                                  variant="determinate"
                                  value={planet.confidence * 100}
                                  sx={{
                                    width: 100,
                                    height: 8,
                                    borderRadius: 4,
                                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                    '& .MuiLinearProgress-bar': {
                                      backgroundColor: getConfidenceColor(planet.confidence),
                                      borderRadius: 4,
                                    },
                                  }}
                                />
                                <Typography variant="body2" className="text-white">
                                  {(planet.confidence * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <IconButton
                                onClick={() => handleView3D(planet)}
                                sx={{
                                  color: '#60a5fa',
                                  '&:hover': {
                                    backgroundColor: 'rgba(96, 165, 250, 0.1)'
                                  }
                                }}
                              >
                                <VisibilityIcon />
                              </IconButton>
                            </TableCell>
                            <TableCell>
                              <IconButton
                                onClick={() => handleExpandRow(planet.id)}
                                sx={{
                                  color: 'white',
                                  '&:hover': {
                                    backgroundColor: 'rgba(255, 255, 255, 0.1)'
                                  }
                                }}
                              >
                                {expandedRows[planet.id] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                              </IconButton>
                            </TableCell>
                          </TableRow>
                          {expandedRows[planet.id] && (
                            <TableRow>
                              <TableCell colSpan={5} className="bg-gray-800/30 p-6">
                                <Typography variant="h6" className="text-white font-semibold mb-4">
                                  Prediction Reasoning:
                                </Typography>
                                {planet.reasons && planet.reasons.length > 0 ? (
                                  <ul className="list-disc list-inside space-y-2">
                                    {planet.reasons.map((reason, index) => (
                                      <li key={index} className="text-gray-300">
                                        {reason}
                                      </li>
                                    ))}
                                  </ul>
                                ) : (
                                  <Typography variant="body2" className="text-gray-300">
                                    Feature importance data available in model output.
                                  </Typography>
                                )}
                              </TableCell>
                            </TableRow>
                          )}
                        </React.Fragment>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>

            {/* 底部操作按鈕 */}
            <div className="flex justify-center gap-6 mb-10">
              <Button
                variant="outlined"
                size="large"
                onClick={() => navigate("/pretrained")}
                sx={{
                  color: '#f5eff7',
                  borderColor: '#f5eff7',
                  px: 4,
                  py: 2,
                  fontSize: '1.125rem',
                  '&:hover': {
                    borderColor: '#f5eff7',
                    backgroundColor: 'rgba(245, 239, 247, 0.1)'
                  }
                }}
              >
                New Prediction
              </Button>
            </div>
          </>
        )}
      </main>

      {/* 3D 可視化對話框 */}
      <Dialog
        open={openDialog}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            backgroundColor: 'rgba(20, 23, 30, 0.95)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '16px'
          }
        }}
      >
        <DialogTitle className="text-white font-semibold">
          3D Visualization - {selectedPlanet?.name}
        </DialogTitle>
        <DialogContent>
          <div className="h-96 bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-lg border border-gray-600/30 overflow-hidden">
            {selectedPlanet && selectedPlanet.prediction === 'Exoplanet' ? (
              <ExoplanetVisualization width={800} height={384} />
            ) : (
              <StarVisualization width={800} height={384} temperature={selectedPlanet?.temperature || 40000} />
            )}
          </div>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={handleCloseDialog}
            sx={{
              color: 'white',
              '&:hover': {
                backgroundColor: 'rgba(255, 255, 255, 0.1)'
              }
            }}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}

export default PretrainedResultPage;
