import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import CSSBackgroundVisualization from "../../components/CSSBackgroundVisualization";
import SimpleTrailCursor from "../../components/SimpleTrailCursor";
import TravelExploreIcon from '@mui/icons-material/TravelExplore';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DownloadIcon from '@mui/icons-material/Download';
import AutoAwesome from '@mui/icons-material/AutoAwesome';
import AutoFixHigh from '@mui/icons-material/AutoFixHigh';
import { 
  Button, 
  Card, 
  CardContent, 
  Typography, 
} from '@mui/material';

function HomePage() {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState(null);

  // 數據格式選項
  const dataFormats = [
    { 
      id: 'kepler', 
      name: 'Kepler Dataset', 
      description: 'Kepler Space Telescope Data',
      img: '/kepler.png',
      color: '#fbbf24'
    },
    { 
      id: 'k2', 
      name: 'K2 Dataset', 
      description: 'K2 Mission Data',
      img: '/k2.png',
      color: '#ef4444'
    },
    { 
      id: 'tess', 
      name: 'TESS Dataset', 
      description: 'TESS Mission Data',
      img: '/tess.png',
      color: '#3b82f6'
    }
  ];

  const handleFileUpload = (file) => {
    if (file && file.type === 'text/csv') {
      setUploadedFile(file);
    } else {
      alert('Please upload a CSV file');
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleDownloadSample = (formatId) => {
    const link = document.createElement('a');
    link.href = '/sample.csv';
    link.download = `${formatId}_sample.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const canStartPrediction = () => {
    return uploadedFile !== null && selectedFormat !== null;
  };

  const handleStartPrediction = () => {
    if (canStartPrediction()) {
      // 直接導航到 PretrainedResultPage
      navigate("/pretrained_result");
    }
  };

  return (
    <div className="relative w-full min-h-screen">
      <CSSBackgroundVisualization />
      <SimpleTrailCursor />
      <Navbar />

      <main className="relative z-10 px-[3.375rem] pt-32 mx-10">
        <h1 className="font-bold text-white text-5xl mb-10 max-w-4xl mx-auto text-left">
          Introduction
        </h1>

        <p className="text-white text-xl leading-relaxed mb-10 max-w-4xl mx-auto text-left">
          Welcome to the Exoplanet AI Playground.
          <br />
          Here, you can explore how scientists search for worlds beyond our
          solar system. Learn about exoplanets, experiment with NASA data or
          your own dataset, and train machine learning models to classify new
          worlds. Adjust parameters, test predictions, and visualize results in
          3D.
        </p>

        {/* Quick Prediction Section */}
        <Card 
          sx={{
            backgroundColor: 'rgba(0, 0, 0, 0.1)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '16px',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
          }}
          className="mb-10 max-w-4xl mx-auto"
        >
          <CardContent className="p-8">
            <Typography variant="h4" className="text-white font-bold mb-6 text-center">
              Predict
            </Typography>
            <Typography variant="body1" className="text-gray-300 mb-8 text-center">
              Upload your testing data and get instant predictions using our pre-trained model
            </Typography>

            {/* Data Format Selection */}
            <div className="mb-8">
              <Typography variant="h6" className="text-white font-semibold mb-4">
                Select Data Format
              </Typography>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {dataFormats.map((format) => (
                  <Card
                    key={format.id}
                    className={`cursor-pointer transition-all duration-300 overflow-hidden ${
                      selectedFormat === format.id 
                        ? 'ring-2 ring-blue-400 shadow-lg shadow-blue-500/20' 
                        : 'hover:shadow-lg hover:shadow-gray-500/10'
                    }`}
                    onClick={() => setSelectedFormat(format.id)}
                    sx={{
                      cursor: 'pointer',
                      backgroundColor: 'rgba(0, 0, 0, 0.1)',
                      backdropFilter: 'blur(10px)',
                      border: selectedFormat === format.id 
                        ? '3px solid #60a5fa' 
                        : '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '16px',
                      backgroundImage: `url(${format.img})`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center',
                      backgroundRepeat: 'no-repeat',
                      aspectRatio: '4/3',
                      position: 'relative',
                      '&:hover': {
                        transform: 'translateY(-2px) scale(1.02)',
                        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)',
                      }
                    }}
                  >
                    {/* 漸層遮罩 */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                    
                    {/* 下載按鈕 - 右上角 */}
                    <div className="absolute top-4 right-4 z-20">
                      <Button
                        size="small"
                        startIcon={<DownloadIcon />}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownloadSample(format.id);
                        }}
                        sx={{
                          color: 'white',
                          backgroundColor: 'rgba(96, 165, 250, 0.8)',
                          fontSize: '0.75rem',
                          minWidth: 'auto',
                          padding: '4px 8px',
                          '&:hover': {
                            backgroundColor: 'rgba(96, 165, 250, 1)',
                          }
                        }}
                      >
                        Sample
                      </Button>
                    </div>
                    
                    {/* 文字內容 */}
                    <CardContent className="absolute bottom-0 left-0 right-0 p-4 text-center z-10">
                      <Typography variant="h6" className="text-white font-bold text-lg drop-shadow-lg">
                        {format.name}
                      </Typography>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            {/* File Upload */}
            <div className="mb-8">
              <Typography variant="h6" className="text-white font-semibold mb-4">
                Upload Testing Data
              </Typography>
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
                  dragActive
                    ? 'border-blue-400 bg-blue-500/10'
                    : uploadedFile
                    ? 'border-green-400 bg-green-500/10'
                    : 'border-gray-600 bg-gray-700/30 hover:border-gray-500'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <CloudUploadIcon 
                  sx={{ 
                    fontSize: 48, 
                    color: uploadedFile ? '#10b981' : '#9ca3af',
                    mb: 2 
                  }} 
                />
                <Typography variant="h6" className="text-white mb-2">
                  {uploadedFile ? uploadedFile.name : 'Drag & drop your CSV file here'}
                </Typography>
                <Typography variant="body2" className="text-gray-400 mb-4">
                  or click to browse files
                </Typography>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileInput}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload">
                  <Button
                    variant="outlined"
                    component="span"
                    sx={{
                      color: 'white',
                      borderColor: 'rgba(255, 255, 255, 0.3)',
                      '&:hover': {
                        borderColor: 'white',
                        backgroundColor: 'rgba(255, 255, 255, 0.1)'
                      }
                    }}
                  >
                    Choose File
                  </Button>
                </label>
              </div>
            </div>

            {/* Start Prediction Button */}
            <div className="text-center">
              <Button
                variant="contained"
                size="large"
                onClick={handleStartPrediction}
                disabled={!canStartPrediction()}
                startIcon={<PlayArrowIcon />}
                sx={{
                  backgroundColor: canStartPrediction() ? '#2563eb' : '#6b7280',
                  '&:hover': {
                    backgroundColor: canStartPrediction() ? '#1d4ed8' : '#6b7280'
                  },
                  px: 6,
                  py: 2,
                  fontSize: '1.125rem'
                }}
              >
                Start Prediction
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-center gap-10 mb-10">

          <Button
            variant="outlined"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            startIcon={<TravelExploreIcon />}
            onClick={() => navigate("/learn_exo")}
            sx={{
              color: '#f5eff7',
              borderColor: '#f5eff7',
              '&:hover': {
                borderColor: '#f5eff7',
                backgroundColor: 'rgba(245, 239, 247, 0.1)'
              }
            }}
          >
            Learn about Exoplanet
          </Button>

          <Button
            variant="outlined"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            startIcon={<PsychologyIcon />}
            onClick={() => navigate("/learn_ml")}
            sx={{
              color: '#f5eff7',
              borderColor: '#f5eff7',
              '&:hover': {
                borderColor: '#f5eff7',
                backgroundColor: 'rgba(245, 239, 247, 0.1)'
              }
            }}
          >
            Learn about ML
          </Button>

          <Button
            variant="contained"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            startIcon={<AutoFixHigh />}
            onClick={() => navigate("/custom")}
            sx={{
              backgroundColor: '#921aff',
              '&:hover': {
                backgroundColor: '#b15bff'
              },
              color: 'white'
            }}
          >
            Train Your AI
          </Button>

          <Button
            variant="contained"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            startIcon={<AutoAwesome />}
            onClick={() => navigate("/pretrained")}
          >
            Using Pretrained Model
          </Button>

        </div>
      </main>
    </div>
  );
}

export default HomePage;
