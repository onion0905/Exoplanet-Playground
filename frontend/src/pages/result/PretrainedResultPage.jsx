import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import ExoplanetVisualization from "../../components/ExoplanetVisualization";
import StarVisualization from "../../components/StarVisualization";
import ConfusionMatrix from "../../components/ConfusionMatrix";
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

function PretrainedResultPage() {
  const navigate = useNavigate();
  const [expandedRows, setExpandedRows] = useState({});
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedPlanet, setSelectedPlanet] = useState(null);

  // 模擬預測結果數據 - 三分類
  const predictionResults = [
    {
      id: 1,
      name: "Kepler-452b",
      prediction: "Exoplanet",
      confidence: 0.997,
      reasons: [
        "Strong transit signal detected with 0.8% depth",
        "Orbital period of 385 days indicates stable orbit",
        "Stellar radius of 1.11 solar radii suggests habitable zone",
        "Transit duration of 10.6 hours consistent with exoplanet"
      ],
      classificationType: "exoplanet"
    },
    {
      id: 2,
      name: "Kepler-186f",
      prediction: "Exoplanet",
      confidence: 0.997,
      reasons: [
        "Clear transit pattern with 0.3% depth",
        "Orbital period of 130 days in habitable zone",
        "M-dwarf host star with stable light curve",
        "No stellar activity correlation detected"
      ],
      classificationType: "exoplanet"
    },
    {
      id: 3,
      name: "Kepler-1234",
      prediction: "Candidate",
      confidence: 0.997,
      reasons: [
        "Transit signal shows some planetary characteristics",
        "Moderate transit depth with periodic pattern",
        "Requires further observation to confirm",
        "Potential stellar activity interference"
      ],
      classificationType: "candidate"
    },
    {
      id: 4,
      name: "Kepler-5678",
      prediction: "Exoplanet",
      confidence: 0.997,
      reasons: [
        "Deep transit signal of 1.2% depth",
        "Circular orbit with 200-day period",
        "Consistent transit timing variations",
        "Secondary eclipse detected confirming planetary nature"
      ],
      classificationType: "exoplanet"
    },
    {
      id: 5,
      name: "Kepler-9999",
      prediction: "Not-Exoplanet",
      confidence: 0.997,
      reasons: [
        "Transit depth varies with stellar activity cycle",
        "Light curve shows stellar pulsation patterns",
        "No clear orbital periodicity",
        "Background binary star contamination detected"
      ],
      classificationType: "not-exoplanet"
    },
    {
      id: 6,
      name: "Kepler-2468",
      prediction: "Candidate",
      confidence: 0.997,
      reasons: [
        "Weak but consistent transit signal",
        "Uncertain orbital parameters",
        "Possible instrumental noise",
        "Needs additional data for confirmation"
      ],
      classificationType: "candidate"
    },
    {
      id: 7,
      name: "Kepler-1357",
      prediction: "Not-Exoplanet",
      confidence: 0.997,
      reasons: [
        "Transit signal correlates with stellar rotation",
        "Asymmetric light curve indicates stellar spot",
        "No secondary eclipse detected",
        "Stellar variability confirmed"
      ],
      classificationType: "not-exoplanet"
    }
  ];

  // 混淆矩陣數據
  const confusionMatrixData = [
    [140, 2, 1],   // Not-Exoplanet: 140正確, 2被誤判為Candidate, 1被誤判為Exoplanet
    [1, 58, 1],    // Candidate: 1被誤判為Not-Exoplanet, 58正確, 1被誤判為Exoplanet
    [1, 2, 170]    // Exoplanet: 1被誤判為Not-Exoplanet, 2被誤判為Candidate, 170正確
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
    switch (prediction) {
      case 'Exoplanet':
        return '#10b981'; // green
      case 'Candidate':
        return '#f59e0b'; // amber
      case 'Not-Exoplanet':
        return '#ef4444'; // red
      default:
        return '#6b7280'; // gray
    }
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
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-gradient-to-br from-green-800/60 to-green-900/60 backdrop-blur-sm border border-green-600/30">
            <CardContent className="text-center p-6">
              <Typography variant="h4" className="text-green-400 font-bold mb-2">
                {predictionResults.filter(p => p.prediction === 'Exoplanet').length}
              </Typography>
              <Typography variant="body1" className="text-white">
                Exoplanets
              </Typography>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-amber-800/60 to-amber-900/60 backdrop-blur-sm border border-amber-600/30">
            <CardContent className="text-center p-6">
              <Typography variant="h4" className="text-amber-400 font-bold mb-2">
                {predictionResults.filter(p => p.prediction === 'Candidate').length}
              </Typography>
              <Typography variant="body1" className="text-white">
                Candidates
              </Typography>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-red-800/60 to-red-900/60 backdrop-blur-sm border border-red-600/30">
            <CardContent className="text-center p-6">
              <Typography variant="h4" className="text-red-400 font-bold mb-2">
                {predictionResults.filter(p => p.prediction === 'Not-Exoplanet').length}
              </Typography>
              <Typography variant="body1" className="text-white">
                Not-Exoplanets
              </Typography>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-blue-800/60 to-blue-900/60 backdrop-blur-sm border border-blue-600/30">
            <CardContent className="text-center p-6">
              <Typography variant="h4" className="text-blue-400 font-bold mb-2">
                {(predictionResults.reduce((acc, p) => acc + p.confidence, 0) / predictionResults.length * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body1" className="text-white">
                Avg Confidence
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
                    onClick={() => navigate("/select")}
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

        {/* Confusion Matrix */}
        <Card className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 shadow-2xl mb-8">
          <CardContent className="p-0">
            <div className="p-6 border-b border-gray-600/30">
              <Typography variant="h5" className="text-white font-semibold">
                Model Performance Analysis
              </Typography>
              <Typography variant="body2" className="text-gray-400 mt-2">
                Confusion Matrix showing classification accuracy across all three categories
              </Typography>
            </div>
            <div className="p-6">
              <ConfusionMatrix 
                data={confusionMatrixData}
                labels={["Not-Exoplanet", "Candidate", "Exoplanet"]}
                title="Pretrained Model Performance"
                className="min-h-0"
              />
            </div>
          </CardContent>
        </Card>

        {/* 底部操作按鈕 */}
        <div className="flex justify-center gap-6 mb-10">
          <Button
            variant="outlined"
            size="large"
            onClick={() => navigate("/select")}
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
          <div className="w-full aspect-[2/1] bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-lg border border-gray-600/30 overflow-hidden">
            {selectedPlanet?.classificationType === 'exoplanet' ? (
              <ExoplanetVisualization width={800} height={400} />
            ) : selectedPlanet?.classificationType === 'candidate' ? (
              <div className="w-full h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-amber-500/20 to-amber-600/20 rounded-full flex items-center justify-center">
                    <span className="text-6xl">🔍</span>
                  </div>
                  <p className="text-white text-lg font-semibold">Candidate Object</p>
                  <p className="text-gray-400 text-sm">Requires further observation</p>
                </div>
              </div>
            ) : (
              <StarVisualization width={800} height={400} temperature={selectedPlanet?.temperature || 40000} />
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
