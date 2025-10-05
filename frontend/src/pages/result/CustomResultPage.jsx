import React, { useState } from "react";
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
  Box,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Card,
  CardContent
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import AddIcon from '@mui/icons-material/Add';
import DownloadIcon from '@mui/icons-material/Download';
import VisibilityIcon from '@mui/icons-material/Visibility';

function CustomResultPage() {
  const navigate = useNavigate();
  const [expandedRows, setExpandedRows] = useState({});
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedPlanet, setSelectedPlanet] = useState(null);

  // 模擬預測結果數據
  const predictionResults = [
    {
      id: 1,
      name: "Kepler-452b",
      prediction: "Exoplanet",
      confidence: 0.94,
      reasons: [
        "Strong transit signal detected with 0.8% depth",
        "Orbital period of 385 days indicates stable orbit",
        "Stellar radius of 1.11 solar radii suggests habitable zone",
        "Transit duration of 10.6 hours consistent with exoplanet"
      ],
      isPositive: true
    },
    {
      id: 2,
      name: "Kepler-186f",
      prediction: "Exoplanet",
      confidence: 0.87,
      reasons: [
        "Clear transit pattern with 0.3% depth",
        "Orbital period of 130 days in habitable zone",
        "M-dwarf host star with stable light curve",
        "No stellar activity correlation detected"
      ],
      isPositive: true
    },
    {
      id: 3,
      name: "Kepler-1234",
      prediction: "False Positive",
      confidence: 0.76,
      reasons: [
        "Transit signal correlates with stellar rotation period",
        "Variable transit depth suggests stellar activity",
        "Asymmetric transit shape indicates stellar spot",
        "No secondary eclipse detected"
      ],
      isPositive: false
    },
    {
      id: 4,
      name: "Kepler-5678",
      prediction: "Exoplanet",
      confidence: 0.91,
      reasons: [
        "Deep transit signal of 1.2% depth",
        "Circular orbit with 200-day period",
        "Consistent transit timing variations",
        "Secondary eclipse detected confirming planetary nature"
      ],
      isPositive: true
    },
    {
      id: 5,
      name: "Kepler-9999",
      prediction: "False Positive",
      confidence: 0.82,
      reasons: [
        "Transit depth varies with stellar activity cycle",
        "Light curve shows stellar pulsation patterns",
        "No clear orbital periodicity",
        "Background binary star contamination detected"
      ],
      isPositive: false
    }
  ];

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
    // 模擬下載功能
    const csvContent = "data:text/csv;charset=utf-8," + 
      "Planet Name,Prediction,Confidence\n" +
      predictionResults.map(planet => 
        `${planet.name},${planet.prediction},${planet.confidence}`
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
    return prediction === 'Exoplanet' ? '#10b981' : '#ef4444';
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

        {/* 結果統計卡片 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="bg-gradient-to-br from-green-800/60 to-green-900/60 backdrop-blur-sm border border-green-600/30">
            <CardContent className="text-center p-6">
              <Typography variant="h4" className="text-green-400 font-bold mb-2">
                {predictionResults.filter(p => p.prediction === 'Exoplanet').length}
              </Typography>
              <Typography variant="body1" className="text-white">
                Exoplanets Detected
              </Typography>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-red-800/60 to-red-900/60 backdrop-blur-sm border border-red-600/30">
            <CardContent className="text-center p-6">
              <Typography variant="h4" className="text-red-400 font-bold mb-2">
                {predictionResults.filter(p => p.prediction === 'False Positive').length}
              </Typography>
              <Typography variant="body1" className="text-white">
                False Positives
              </Typography>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-blue-800/60 to-blue-900/60 backdrop-blur-sm border border-blue-600/30">
            <CardContent className="text-center p-6">
              <Typography variant="h4" className="text-blue-400 font-bold mb-2">
                {(predictionResults.reduce((acc, p) => acc + p.confidence, 0) / predictionResults.length * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body1" className="text-white">
                Average Confidence
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
                    onClick={() => navigate("/custom")}
                    sx={{
                      color: 'white',
                      borderColor: 'rgba(255, 255, 255, 0.3)',
                      '&:hover': {
                        borderColor: 'white',
                        backgroundColor: 'rgba(255, 255, 255, 0.1)'
                      }
                    }}
                  >
                    Train New Model
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
                  {predictionResults.map((planet) => (
                    <React.Fragment key={planet.id}>
                      <TableRow className="hover:bg-gray-700/20">
                        <TableCell className="text-white font-medium">
                          {planet.name}
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
                          {planet.isPositive && (
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
                          )}
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
                            <ul className="list-disc list-inside space-y-2">
                              {planet.reasons.map((reason, index) => (
                                <li key={index} className="text-gray-300">
                                  {reason}
                                </li>
                              ))}
                            </ul>
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
            onClick={() => navigate("/custom")}
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
            Train New Model
          </Button>
        </div>
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
          <div className="h-96 bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-lg flex items-center justify-center border border-gray-600/30">
            <div className="text-center">
              <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center">
                <span className="text-6xl">🌍</span>
              </div>
              <Typography variant="h6" className="text-white mb-2">
                3D Planet Visualization
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                Interactive 3D model of {selectedPlanet?.name}
              </Typography>
              <Typography variant="body2" className="text-gray-400 mt-2">
                (This would be a real 3D visualization in production)
              </Typography>
            </div>
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

export default CustomResultPage;
