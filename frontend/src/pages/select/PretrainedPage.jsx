import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { 
  Button,
  Card,
  CardContent,
  Typography
} from '@mui/material';

function PretrainedPage() {
  const navigate = useNavigate();
  const [dataSource, setDataSource] = useState('nasa'); // 'nasa' or 'user'
  const [selectedDataset, setSelectedDataset] = useState('kepler');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const datasets = [
    {
      id: 'kepler',
      img: '/kepler.png',
      name: 'Kepler Dataset',
      color: '#fbbf24'
    },
    {
      id: 'k2',
      img: '/k2.png',
      name: 'K2 Dataset', 
      color: '#ef4444'
    },
    {
      id: 'tess',
      img: '/tess.png',
      name: 'TESS Dataset',
      color: '#3b82f6'
    }
  ];

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setUploadedFile(file);
    }
  };

  const handleDataSourceChange = (source) => {
    setDataSource(source);
    if (source === 'nasa') {
      setUploadedFile(null);
      // 重置為第一個 NASA 數據集
      setSelectedDataset('kepler');
    } else {
      setSelectedDataset('');
    }
  };

  const handleDatasetSelect = (dataset) => {
    setSelectedDataset(dataset);
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
      const file = e.dataTransfer.files[0];
      if (file.type === 'text/csv') {
        setUploadedFile(file);
      }
    }
  };

  const handleStartPrediction = () => {
    navigate("/pretrained_result");
  };

  // 檢查是否可以開始預測
  const canStartPrediction = () => {
    return (dataSource === 'nasa' && selectedDataset) || (dataSource === 'user' && uploadedFile);
  };

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img
        className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover"
        alt="Space background"
        src="/background.svg"
      />

      <Navbar />

      <main className="relative z-10 px-[3.375rem] pt-32 max-w-6xl mx-auto">
        <h1 className="font-bold text-white text-5xl mb-10 text-left mx-auto">
          Use Pretrained Model
        </h1>
        
        <p className="text-white text-lg leading-relaxed mb-8 text-left max-w-4xl mx-auto">
          Select your data to use with our pretrained model for exoplanet detection.
        </p>
        
        {/* Data Source Selection */}
        <div className="mb-8">
          <h2 className="text-white text-2xl font-semibold mb-6">Select your data</h2>
          
          <div className="mb-6">
            <div className="flex gap-4 mb-6">
              <div 
                className={`flex items-center p-4 rounded-lg cursor-pointer transition-all duration-200 ${
                  dataSource === 'nasa' 
                    ? 'bg-blue-500/20 border-2 border-blue-500' 
                    : 'bg-gray-800/50 border-2 border-gray-600 hover:bg-gray-700/50'
                }`}
                onClick={() => handleDataSourceChange('nasa')}
              >
                <div className={`w-4 h-4 rounded-full border-2 mr-3 ${
                  dataSource === 'nasa' 
                    ? 'border-blue-500 bg-blue-500' 
                    : 'border-gray-400'
                }`}>
                  {dataSource === 'nasa' && (
                    <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                  )}
                </div>
                <div>
                  <Typography variant="h6" className="text-white font-semibold">
                    NASA Datasets
                  </Typography>
                  <Typography variant="body2" className="text-gray-300">
                    Use pre-processed NASA exoplanet data
                  </Typography>
                </div>
              </div>

              <div 
                className={`flex items-center p-4 rounded-lg cursor-pointer transition-all duration-200 ${
                  dataSource === 'user' 
                    ? 'bg-blue-500/20 border-2 border-blue-500' 
                    : 'bg-gray-800/50 border-2 border-gray-600 hover:bg-gray-700/50'
                }`}
                onClick={() => handleDataSourceChange('user')}
              >
                <div className={`w-4 h-4 rounded-full border-2 mr-3 ${
                  dataSource === 'user' 
                    ? 'border-blue-500 bg-blue-500' 
                    : 'border-gray-400'
                }`}>
                  {dataSource === 'user' && (
                    <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                  )}
                </div>
                <div>
                  <Typography variant="h6" className="text-white font-semibold">
                    Upload Your Data
                  </Typography>
                  <Typography variant="body2" className="text-gray-300">
                    Upload your own CSV file
                  </Typography>
                </div>
              </div>
            </div>

            {/* NASA Dataset Selection */}
            {dataSource === 'nasa' && (
              <div className="p-8 rounded-2xl bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 shadow-2xl">
                <div className="flex items-center mb-6">
                  <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center mr-3">
                    <span className="text-blue-400 text-lg">🛰️</span>
                  </div>
                  <h3 className="text-white text-xl font-semibold">Choose NASA Dataset</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {datasets.map((dataset) => {
                    return (
                      <Card
                        key={dataset.id}
                        className={`cursor-pointer transition-all duration-300 overflow-hidden ${
                          selectedDataset === dataset.id 
                            ? 'ring-2 ring-blue-400 shadow-lg shadow-blue-500/20' 
                            : 'hover:shadow-lg hover:shadow-gray-500/10'
                        }`}
                        onClick={() => handleDatasetSelect(dataset.id)}
                        sx={{
                          cursor: 'pointer',
                          backdropFilter: 'blur(10px)',
                          border: selectedDataset === dataset.id 
                            ? '3px solid #60a5fa' 
                            : '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: '16px',
                          backgroundImage: `url(${dataset.img})`,
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
                        
                        {/* 文字內容 */}
                        <CardContent className="absolute bottom-0 left-0 right-0 p-6 text-center z-10">
                          <Typography variant="h6" className="text-white font-bold text-lg drop-shadow-lg">
                            {dataset.name}
                          </Typography>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            )}

            {/* User Data Upload */}
            {dataSource === 'user' && (
              <div className="p-8 rounded-2xl bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 shadow-2xl">
                <div className="flex items-center mb-6">
                  <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center mr-3">
                    <span className="text-green-400 text-lg">📁</span>
                  </div>
                  <h3 className="text-white text-xl font-semibold">Upload Your Data</h3>
                </div>
                
                <div
                  className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 backdrop-blur-sm ${
                    dragActive 
                      ? 'border-blue-400 bg-gradient-to-br from-blue-500/20 to-blue-600/10 shadow-lg shadow-blue-500/20' 
                      : 'border-gray-500/50 bg-gradient-to-br from-gray-700/30 to-gray-800/30 hover:border-gray-400 hover:bg-gradient-to-br hover:from-gray-600/40 hover:to-gray-700/40'
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-gray-600/50 to-gray-700/50 rounded-2xl flex items-center justify-center backdrop-blur-sm border border-gray-500/30">
                    <CloudUploadIcon className="text-3xl text-gray-300" />
                  </div>
                  <p className="text-white mb-4 text-lg font-medium">Drag and drop your CSV file here</p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
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
                        backgroundColor: 'rgba(255, 255, 255, 0.05)',
                        backdropFilter: 'blur(10px)',
                        px: 4,
                        py: 1.5,
                        fontSize: '1rem',
                        fontWeight: 500,
                        borderRadius: '12px',
                        '&:hover': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          transform: 'translateY(-1px)',
                          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
                        }
                      }}
                    >
                      Choose File
                    </Button>
                  </label>
                  <p className="text-gray-400 text-sm mt-4">
                    Supported format: CSV files with exoplanet data
                  </p>
                  {uploadedFile && (
                    <div className="mt-6 p-4 bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-400/30 rounded-xl backdrop-blur-sm">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center mr-3">
                            <span className="text-white text-xs">✓</span>
                          </div>
                          <span className="text-green-200 text-sm font-medium">{uploadedFile.name}</span>
                        </div>
                        <Button
                          size="small"
                          onClick={() => setUploadedFile(null)}
                          sx={{
                            color: 'rgba(255, 255, 255, 0.7)',
                            minWidth: 'auto',
                            padding: '4px 8px',
                            '&:hover': {
                              color: 'white',
                              backgroundColor: 'rgba(255, 255, 255, 0.1)'
                            }
                          }}
                        >
                          ✕
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Start Prediction Button */}
        <div className="text-center mt-8 mb-10">
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrowIcon />}
            onClick={handleStartPrediction}
            disabled={!canStartPrediction()}
            sx={{
              backgroundColor: '#2563eb',
              '&:hover': {
                backgroundColor: '#1d4ed8'
              },
              '&:disabled': {
                backgroundColor: '#6b7280'
              },
              color: 'white',
              px: 4,
              py: 2,
              fontSize: '1.125rem'
            }}
          >
            Start Prediction!
          </Button>
        </div>

      </main>
    </div>
  );
}

export default PretrainedPage;