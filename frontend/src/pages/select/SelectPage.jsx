import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import apiService from "../../lib/api";
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { 
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Card,
  CardContent,
  Typography,
  Chip
} from '@mui/material';

function SelectPage() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [dataSource, setDataSource] = useState('nasa'); // 'nasa' or 'user'
  const [selectedDataset, setSelectedDataset] = useState('kepler');
  const [selectedModel, setSelectedModel] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [hyperparameters, setHyperparameters] = useState({});
  
  // Backend integration state
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendDatasets, setBackendDatasets] = useState({});
  const [backendModels, setBackendModels] = useState([]);
  const [uploadedFileInfo, setUploadedFileInfo] = useState(null);
  const [maxStep, setMaxStep] = useState(1);
  const [isTrainingReady, setIsTrainingReady] = useState(false);

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

  const models = [
    { value: 'linear_regression', label: 'Linear Regression' },
    { value: 'svm', label: 'Support Vector Machine (SVM)' },
    { value: 'decision_tree', label: 'Decision Tree' },
    { value: 'random_forest', label: 'Random Forest' },
    { value: 'xgboost', label: 'XGBoost' },
    { value: 'pca', label: 'Principal Component Analysis (PCA)' },
    { value: 'neural_network', label: 'Neural Network' }
  ];

  const getHyperparameters = (model) => {
    const params = {
      linear_regression: [
        { name: 'fit_intercept', label: 'Fit Intercept', type: 'select', options: ['true', 'false'], default: 'true' },
        { name: 'normalize', label: 'Normalize', type: 'select', options: ['false', 'true'], default: 'false' }
      ],
      svm: [
        { name: 'C', label: 'C (Regularization)', type: 'number', default: 1.0, step: 0.1, min: 0.1 },
        { name: 'kernel', label: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], default: 'rbf' },
        { name: 'gamma', label: 'Gamma', type: 'select', options: ['scale', 'auto'], default: 'scale' }
      ],
      decision_tree: [
        { name: 'max_depth', label: 'Max Depth', type: 'number', placeholder: 'None (unlimited)', min: 1 },
        { name: 'min_samples_split', label: 'Min Samples Split', type: 'number', default: 2, min: 2 },
        { name: 'min_samples_leaf', label: 'Min Samples Leaf', type: 'number', default: 1, min: 1 },
        { name: 'criterion', label: 'Criterion', type: 'select', options: ['gini', 'entropy'], default: 'gini' }
      ],
      random_forest: [
        { name: 'n_estimators', label: 'Number of Estimators', type: 'number', default: 100, min: 1 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', placeholder: 'None (unlimited)', min: 1 },
        { name: 'min_samples_split', label: 'Min Samples Split', type: 'number', default: 2, min: 2 },
        { name: 'min_samples_leaf', label: 'Min Samples Leaf', type: 'number', default: 1, min: 1 }
      ],
      xgboost: [
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.1, step: 0.01, min: 0.01, max: 1 },
        { name: 'max_depth', label: 'Max Depth', type: 'number', default: 6, min: 1 },
        { name: 'n_estimators', label: 'N Estimators', type: 'number', default: 100, min: 1 },
        { name: 'subsample', label: 'Subsample', type: 'number', default: 1.0, step: 0.1, min: 0.1, max: 1 }
      ],
      pca: [
        { name: 'n_components', label: 'Number of Components', type: 'number', placeholder: 'Auto (min of features/samples)', min: 1 },
        { name: 'svd_solver', label: 'SVD Solver', type: 'select', options: ['auto', 'full', 'arpack', 'randomized'], default: 'auto' }
      ],
      neural_network: [
        { name: 'hidden_layer_sizes', label: 'Hidden Layer Sizes (comma-separated)', type: 'text', default: '100', placeholder: 'e.g., 100,50,25' },
        { name: 'activation', label: 'Activation Function', type: 'select', options: ['relu', 'tanh', 'logistic'], default: 'relu' },
        { name: 'learning_rate', label: 'Learning Rate', type: 'number', default: 0.001, step: 0.0001, min: 0.0001 },
        { name: 'max_iter', label: 'Max Iterations', type: 'number', default: 200, min: 1 }
      ]
    };
    return params[model] || [];
  };

  // Initialize session and load backend data
  useEffect(() => {
    initializeSession();
  }, []);

  // Update computed values when dependencies change
  useEffect(() => {
    const computedMaxStep = (() => {
      if (dataSource === 'nasa' && selectedDataset) return 3;
      if (dataSource === 'user' && uploadedFileInfo) return 3;
      if (dataSource === 'nasa' || dataSource === 'user') return 2;
      return 1;
    })();
    
    const computedIsTrainingReady = selectedModel && 
      ((dataSource === 'nasa' && selectedDataset) || (dataSource === 'user' && uploadedFileInfo)) && 
      currentStep === 3;
      
    setMaxStep(computedMaxStep);
    setIsTrainingReady(computedIsTrainingReady);
  }, [dataSource, selectedDataset, selectedModel, uploadedFileInfo, currentStep]);

  const initializeSession = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.createSession();
      if (response.status === 'success') {
        setSessionId(response.session_id);
        setBackendDatasets(response.datasets || {});
        setBackendModels(response.models || []);
        setCurrentStep(response.current_step || 1);
        setMaxStep(response.max_step || 1);
        setIsTrainingReady(response.is_training_ready || false);
        
        // Set initial values from backend
        const sessionInfo = response.session_info || {};
        if (sessionInfo.data_source) setDataSource(sessionInfo.data_source);
        if (sessionInfo.selected_dataset) setSelectedDataset(sessionInfo.selected_dataset);
        if (sessionInfo.selected_model) setSelectedModel(sessionInfo.selected_model);
        if (sessionInfo.uploaded_file_info) setUploadedFileInfo(sessionInfo.uploaded_file_info);
      }
    } catch (error) {
      console.error('Failed to initialize session:', error);
      setError('Failed to connect to backend. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      try {
        setLoading(true);
        setError(null);
        
        // Upload file to backend
        const response = await apiService.uploadCSV(file, sessionId);
        if (response.status === 'success') {
          setUploadedFile(file);
          setUploadedFileInfo(response.file_info);
          
          // Update backend session
          if (sessionId) {
            await apiService.updateDataSource(sessionId, 'user', null, response.file_info);
          }
        }
      } catch (error) {
        console.error('File upload failed:', error);
        setError(`File upload failed: ${error.message}`);
      } finally {
        setLoading(false);
      }
    } else {
      setError('Please select a valid CSV file.');
    }
  };

  const handleDataSourceChange = async (source) => {
    try {
      setLoading(true);
      setError(null);
      
      setDataSource(source);
      
      if (source === 'nasa') {
        setUploadedFile(null);
        setUploadedFileInfo(null);
        setSelectedDataset('kepler'); // Default to kepler
        
        // Update backend
        if (sessionId) {
          await apiService.updateDataSource(sessionId, 'nasa', 'kepler', null);
        }
      } else {
        setSelectedDataset('');
        
        // Update backend if user has already uploaded a file
        if (sessionId) {
          await apiService.updateDataSource(sessionId, 'user', null, uploadedFileInfo);
        }
      }
    } catch (error) {
      console.error('Failed to update data source:', error);
      setError('Failed to update data source. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSelect = async (dataset) => {
    try {
      setLoading(true);
      setError(null);
      
      setSelectedDataset(dataset);
      
      // Update backend
      if (sessionId) {
        await apiService.updateDataSource(sessionId, 'nasa', dataset, null);
      }
    } catch (error) {
      console.error('Failed to update dataset selection:', error);
      setError('Failed to update dataset. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelect = async (model) => {
    try {
      setLoading(true);
      setError(null);
      
      setSelectedModel(model);
      
      // Update backend
      if (sessionId) {
        await apiService.updateModel(sessionId, model);
      }
    } catch (error) {
      console.error('Failed to update model selection:', error);
      setError('Failed to update model. Please try again.');
    } finally {
      setLoading(false);
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

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'text/csv') {
        try {
          setLoading(true);
          setError(null);
          
          // Upload file to backend
          const response = await apiService.uploadCSV(file, sessionId);
          if (response.status === 'success') {
            setUploadedFile(file);
            setUploadedFileInfo(response.file_info);
            
            // Update backend session
            if (sessionId) {
              await apiService.updateDataSource(sessionId, 'user', null, response.file_info);
            }
          }
        } catch (error) {
          console.error('File upload failed:', error);
          setError(`File upload failed: ${error.message}`);
        } finally {
          setLoading(false);
        }
      } else {
        setError('Please upload a valid CSV file.');
      }
    }
  };

  const handleStartTraining = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Update hyperparameters in backend
      if (sessionId) {
        await apiService.updateHyperparameters(sessionId, hyperparameters);
      }
      
      // Navigate to training with session ID
      navigate(`/training?session_id=${sessionId}`);
    } catch (error) {
      console.error('Failed to prepare for training:', error);
      setError('Failed to prepare training. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // 檢查是否可以進入下一步
  const canGoToNextStep = () => {
    if (currentStep === 1) {
      return (dataSource === 'nasa' && selectedDataset) || (dataSource === 'user' && uploadedFileInfo);
    }
    if (currentStep === 2) {
      return selectedModel !== '';
    }
    return false;
  };

  const handleNextStep = async () => {
    if (canGoToNextStep() && currentStep < maxStep) {
      try {
        setLoading(true);
        setError(null);
        
        // Update step in backend
        if (sessionId) {
          const response = await apiService.nextStep(sessionId);
          if (response.status === 'success') {
            setCurrentStep(response.current_step);
            setMaxStep(response.max_step);
            setIsTrainingReady(response.is_training_ready);
          }
        } else {
          setCurrentStep(currentStep + 1);
        }
      } catch (error) {
        console.error('Failed to proceed to next step:', error);
        setError('Failed to proceed. Please try again.');
      } finally {
        setLoading(false);
      }
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

      <main className="relative z-10 px-[3.375rem] pt-32 max-w-6xl mx-auto">
        <h1 className="font-bold text-white text-5xl mb-10 text-left mx-auto">
          Launch Your Model
        </h1>
        
        <p className="text-white text-lg leading-relaxed mb-8 text-left max-w-4xl mx-auto">
        Here, you can get hands-on experience building and testing your own AI model.
        <br />
        To get started, follow the three steps below:
        </p>

        <ol className="text-white text-lg leading-relaxed mb-8 max-w-4xl mx-auto list-disc list-inside">
        <li>Select your data — Upload your own dataset, or use one of NASA’s existing collections: Kepler Objects of Interest (KOI), TESS Objects of Interest (TOI), or K2 Planets and Candidates.</li>
        <li>Choose a model — Pick a machine learning model to train.</li>
        <li>Set hyperparameters — Adjust key parameters to customize your model’s behavior.</li>
        </ol>

        <p className="text-white text-lg leading-relaxed mb-12 text-left max-w-4xl mx-auto">
          Once you’re ready, hit ‘start training’ and watch your model take off!
        </p>
        

        {/* Progress Indicator */}
        <div className="mb-12">
          <div className="flex items-center justify-center mb-4">
            <div className="flex items-center space-x-8">
              <div 
                className={`flex flex-col items-center w-48 ${
                  maxStep >= 1 ? 'cursor-pointer' : 'cursor-not-allowed'
                }`}
                onClick={() => maxStep >= 1 && setCurrentStep(1)}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold transition-colors ${
                  currentStep >= 1 ? 'bg-blue-500' : 'bg-gray-600'
                }`}>1</div>
                <span className="text-white text-sm mt-2">Select your data</span>
              </div>
              <div className={`w-16 h-1 transition-colors ${
                currentStep >= 2 ? 'bg-blue-500' : 'bg-gray-600'
              }`}></div>
              <div 
                className={`flex flex-col items-center w-48 ${
                  maxStep >= 2 ? 'cursor-pointer' : 'cursor-not-allowed'
                }`}
                onClick={() => maxStep >= 2 && setCurrentStep(2)}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold transition-colors ${
                  currentStep >= 2 ? 'bg-blue-500' : 'bg-gray-600'
                }`}>2</div>
                <span className="text-white text-sm mt-2">Choose a Model</span>
              </div>
              <div className={`w-16 h-1 transition-colors ${
                currentStep >= 3 ? 'bg-blue-500' : 'bg-gray-600'
              }`}></div>
              <div 
                className={`flex flex-col items-center w-48 ${
                  maxStep >= 3 ? 'cursor-pointer' : 'cursor-not-allowed'
                }`}
                onClick={() => maxStep >= 3 && setCurrentStep(3)}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold transition-colors ${
                  currentStep >= 3 ? 'bg-blue-500' : 'bg-gray-600'
                }`}>3</div>
                <span className="text-white text-sm mt-2">Set Hyperparameters</span>
              </div>
            </div>
          </div>
        </div>

        {/* 1. Select your data */}
        {currentStep === 1 && (
          <div className="mb-8">
            <h2 className="text-white text-2xl font-semibold mb-6">1. Select your data</h2>
            
            {/* Data Source Selection */}
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

            {/* Next Button for Step 1 */}
            <div className="text-center mt-8">
              <Button
                variant="contained"
                size="large"
                endIcon={<ArrowForwardIcon />}
                onClick={handleNextStep}
                disabled={!canGoToNextStep()}
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
                Choose a Model
              </Button>
            </div>
          </div>
        )}

        {/* 2. Choose a Model */}
        {currentStep === 2 && (
          <div className="mb-8">
            <h2 className="text-white text-2xl font-semibold mb-6">2. Choose a Model</h2>
            <FormControl fullWidth className="mb-4">
              <InputLabel className="text-white">Choose a machine learning model</InputLabel>
              <Select
                value={selectedModel}
                onChange={(e) => handleModelSelect(e.target.value)}
                className="text-white bg-gray-800/50"
                label="Choose a machine learning model"
              >
                <MenuItem value="">Select a model...</MenuItem>
                {models.map((model) => (
                  <MenuItem key={model.value} value={model.value}>
                    {model.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Next Button for Step 2 */}
            <div className="text-center mt-8">
              <Button
                variant="contained"
                size="large"
                endIcon={<ArrowForwardIcon />}
                onClick={handleNextStep}
                disabled={!canGoToNextStep()}
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
                Set Hyperparameters
              </Button>
            </div>
          </div>
        )}

        {/* 3. Set Hyperparameters */}
        {currentStep === 3 && selectedModel && (
          <div className="mb-8">
            <h2 className="text-white text-2xl font-semibold mb-6">3. Set Hyperparameters</h2>
            <p className="text-gray-300 mb-4">
              Configure your hyperparameter based on your Choose a Model. Default values are provided as a quick start.
            </p>
            
            <div className="flex flex-wrap gap-4">
              {getHyperparameters(selectedModel).map((param) => (
                <div key={param.name} className="flex-1 min-w-[200px]">
                  <label className="block text-white text-sm font-medium mb-2">
                    {param.label}
                  </label>
                  {param.type === 'select' ? (
                    <Select
                      value={hyperparameters[param.name] || param.default || ''}
                      onChange={(e) => setHyperparameters(prev => ({
                        ...prev,
                        [param.name]: e.target.value
                      }))}
                      className="w-full bg-gray-800/50 text-white"
                    >
                      {param.options.map((option) => (
                        <MenuItem key={option} value={option}>
                          {option}
                        </MenuItem>
                      ))}
                    </Select>
                  ) : (
                    <TextField
                      type={param.type}
                      placeholder={param.placeholder}
                      value={hyperparameters[param.name] || param.default || ''}
                      onChange={(e) => setHyperparameters(prev => ({
                        ...prev,
                        [param.name]: e.target.value
                      }))}
                      inputProps={{
                        min: param.min,
                        max: param.max,
                        step: param.step
                      }}
                      className="w-full"
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          color: 'white',
                          '& fieldset': {
                            borderColor: 'gray',
                          },
                          '&:hover fieldset': {
                            borderColor: 'white',
                          },
                        },
                      }}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Start Training Button for Step 3 */}
            <div className="text-center mt-8">
              <Button
                variant="contained"
                size="large"
                startIcon={<PlayArrowIcon />}
                onClick={handleStartTraining}
                disabled={!isTrainingReady}
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
                Start Training!
              </Button>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default SelectPage;