import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import AssessmentIcon from '@mui/icons-material/Assessment';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { 
  Button, 
  Card, 
  CardContent, 
  Typography,
  Box
} from '@mui/material';

function PredictPage() {
  const navigate = useNavigate();
  
  // 模擬訓練時使用的數據源（實際應該從後端獲取）
  const trainingDataSource = 'nasa'; // 或 'user'
  const trainingDataset = 'kepler'; // 實際訓練時使用的數據集
  
  const [testDataSource, setTestDataSource] = useState(trainingDataSource === 'nasa' ? 'nasa' : 'user'); // 'nasa' or 'user'
  const [selectedTestDataset, setSelectedTestDataset] = useState(trainingDataSource === 'nasa' ? trainingDataset : '');
  const [uploadedTestFile, setUploadedTestFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const testDatasets = [
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

  const handleTestDataSourceChange = (source) => {
    setTestDataSource(source);
    if (source === 'nasa') {
      setUploadedTestFile(null);
      setSelectedTestDataset(trainingDataSource === 'nasa' ? trainingDataset : '');
    } else {
      setSelectedTestDataset('');
    }
  };

  const handleTestDatasetSelect = (dataset) => {
    setSelectedTestDataset(dataset);
  };

  const handleTestFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setUploadedTestFile(file);
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
      const file = e.dataTransfer.files[0];
      if (file.type === 'text/csv') {
        setUploadedTestFile(file);
      }
    }
  };

  const handleGetResults = () => {
    navigate("/result");
  };

  // 檢查是否可以獲取結果
  const canGetResults = () => {
    if (testDataSource === 'nasa') {
      return selectedTestDataset !== '';
    } else {
      return uploadedTestFile !== null;
    }
  };

  // 根據訓練數據源限制測試選項
  const getAvailableTestOptions = () => {
    if (trainingDataSource === 'nasa') {
      // NASA 訓練可以使用原先的望遠鏡資料或上傳自身資料
      return {
        allowNasaTest: true,
        allowUserTest: true,
        message: "Since you trained with NASA data, you can test with the same telescope data or upload your own test data."
      };
    } else {
      // 用戶數據訓練不能選擇 NASA 測試數據
      return {
        allowNasaTest: false,
        allowUserTest: true,
        message: "Since you trained with your own data, you can only test with your own data."
      };
    }
  };

  const testOptions = getAvailableTestOptions();

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img
        className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover"
        alt="Space background"
        src="/background.svg"
      />

      <Navbar />

      <main className="relative z-10 px-[3.375rem] pt-32 max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h1 className="font-bold text-white text-5xl mb-6">
            Upload Data for Prediction
          </h1>
          <p className="text-white text-xl mb-8">
            Your model has been trained successfully! Now upload test data to get predictions.
          </p>
        </div>

        {/* 訓練完成狀態 */}
        <div className="bg-gradient-to-br from-green-800/60 to-green-900/60 backdrop-blur-sm border border-green-600/30 rounded-2xl p-6 shadow-2xl mb-8">
          <div className="flex items-center justify-center">
            <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mr-4">
              <span className="text-white text-xl">✓</span>
            </div>
            <div>
              <Typography variant="h6" className="text-white font-semibold">
                Model Training Complete
              </Typography>
              <Typography variant="body2" className="text-green-200">
                Trained with: {trainingDataSource === 'nasa' ? `${trainingDataset} Dataset` : 'Your Custom Data'}
              </Typography>
            </div>
          </div>
        </div>

        {/* 限制說明 */}
        <div className="bg-gradient-to-br from-blue-800/60 to-blue-900/60 backdrop-blur-sm border border-blue-600/30 rounded-2xl p-6 shadow-2xl mb-8">
          <div className="flex items-center">
            <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center mr-3">
              <span className="text-blue-400 text-lg">ℹ️</span>
            </div>
            <Typography variant="body1" className="text-blue-200">
              {testOptions.message}
            </Typography>
          </div>
        </div>

        {/* 測試數據選擇 */}
        <div className="mb-8">
          <h2 className="text-white text-2xl font-semibold mb-6 text-center">Select Test Data</h2>
          
          {/* 數據源選擇 */}
          <div className="mb-6">
            <div className="flex gap-4 mb-6 justify-center">
              <div 
                className={`flex items-center p-4 rounded-lg cursor-pointer transition-all duration-200 ${
                  testDataSource === 'nasa' && testOptions.allowNasaTest
                    ? 'bg-blue-500/20 border-2 border-blue-500' 
                    : testOptions.allowNasaTest
                    ? 'bg-gray-800/50 border-2 border-gray-600 hover:bg-gray-700/50'
                    : 'bg-gray-800/30 border-2 border-gray-700 cursor-not-allowed opacity-50'
                }`}
                onClick={() => testOptions.allowNasaTest && handleTestDataSourceChange('nasa')}
              >
                <div className={`w-4 h-4 rounded-full border-2 mr-3 ${
                  testDataSource === 'nasa' && testOptions.allowNasaTest
                    ? 'border-blue-500 bg-blue-500' 
                    : 'border-gray-400'
                }`}>
                  {testDataSource === 'nasa' && testOptions.allowNasaTest && (
                    <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                  )}
                </div>
                <div>
                  <Typography variant="h6" className="text-white font-semibold">
                    NASA Test Data
                  </Typography>
                  <Typography variant="body2" className="text-gray-300">
                    Use NASA datasets for testing
                  </Typography>
                </div>
              </div>

              <div 
                className={`flex items-center p-4 rounded-lg cursor-pointer transition-all duration-200 ${
                  testDataSource === 'user' && testOptions.allowUserTest
                    ? 'bg-green-500/20 border-2 border-green-500' 
                    : testOptions.allowUserTest
                    ? 'bg-gray-800/50 border-2 border-gray-600 hover:bg-gray-700/50'
                    : 'bg-gray-800/30 border-2 border-gray-700 cursor-not-allowed opacity-50'
                }`}
                onClick={() => testOptions.allowUserTest && handleTestDataSourceChange('user')}
              >
                <div className={`w-4 h-4 rounded-full border-2 mr-3 ${
                  testDataSource === 'user' && testOptions.allowUserTest
                    ? 'border-green-500 bg-green-500' 
                    : 'border-gray-400'
                }`}>
                  {testDataSource === 'user' && testOptions.allowUserTest && (
                    <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                  )}
                </div>
                <div>
                  <Typography variant="h6" className="text-white font-semibold">
                    Upload Test Data
                  </Typography>
                  <Typography variant="body2" className="text-gray-300">
                    Upload your own test data
                  </Typography>
                </div>
              </div>
            </div>

            {/* NASA 測試數據選擇 */}
            {testDataSource === 'nasa' && testOptions.allowNasaTest && (
              <div className="p-8 rounded-2xl bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 shadow-2xl">
                <div className="flex items-center mb-6">
                  <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center mr-3">
                    <span className="text-blue-400 text-lg">🛰️</span>
                  </div>
                  <h3 className="text-white text-xl font-semibold">Use Training Dataset for Testing</h3>
                </div>
                
                {/* 顯示訓練時使用的數據集 */}
                <div className="flex justify-center">
                  <Card
                    className="cursor-pointer transition-all duration-300 overflow-hidden ring-2 ring-blue-400 shadow-lg shadow-blue-500/20"
                    onClick={() => handleTestDatasetSelect(trainingDataset)}
                    sx={{
                      cursor: 'pointer',
                      backdropFilter: 'blur(10px)',
                      border: '3px solid #60a5fa',
                      borderRadius: '16px',
                      backgroundImage: `url(${testDatasets.find(d => d.id === trainingDataset)?.img})`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center',
                      backgroundRepeat: 'no-repeat',
                      aspectRatio: '4/3',
                      position: 'relative',
                      width: '300px',
                      '&:hover': {
                        transform: 'translateY(-2px) scale(1.02)',
                        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)',
                      }
                    }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
                    <CardContent className="absolute bottom-0 left-0 right-0 p-6 text-center z-10">
                      <Typography variant="h6" className="text-white font-bold text-lg drop-shadow-lg">
                        {testDatasets.find(d => d.id === trainingDataset)?.name}
                      </Typography>
                      <Typography variant="body2" className="text-blue-200 text-sm mt-1">
                        Same as training data
                      </Typography>
                    </CardContent>
                  </Card>
                </div>
                
              </div>
            )}

            {/* 用戶測試數據上傳 */}
            {testDataSource === 'user' && testOptions.allowUserTest && (
              <div className="p-8 rounded-2xl bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 shadow-2xl">
                <div className="flex items-center mb-6">
                  <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center mr-3">
                    <span className="text-green-400 text-lg">📁</span>
                  </div>
                  <h3 className="text-white text-xl font-semibold">Upload Your Test Data</h3>
                </div>
                <div
                  className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 backdrop-blur-sm ${
                    dragActive 
                      ? 'border-green-400 bg-gradient-to-br from-green-500/20 to-green-600/10 shadow-lg shadow-green-500/20' 
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
                  <p className="text-white mb-4 text-lg font-medium">Drag and drop your test CSV file here</p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleTestFileUpload}
                    className="hidden"
                    id="test-file-upload"
                  />
                  <label htmlFor="test-file-upload">
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
                  {uploadedTestFile && (
                    <div className="mt-6 p-4 bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-400/30 rounded-xl backdrop-blur-sm">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center mr-3">
                            <span className="text-white text-xs">✓</span>
                          </div>
                          <span className="text-green-200 text-sm font-medium">{uploadedTestFile.name}</span>
                        </div>
                        <Button
                          size="small"
                          onClick={() => setUploadedTestFile(null)}
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

        {/* 獲取結果按鈕 */}
        <div className="text-center mb-10">
          <Button
            variant="contained"
            size="large"
            startIcon={<AssessmentIcon />}
            onClick={handleGetResults}
            disabled={!canGetResults()}
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
            Get Results
          </Button>
        </div>
      </main>
    </div>
  );
}

export default PredictPage;
