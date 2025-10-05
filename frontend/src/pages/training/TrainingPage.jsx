import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import { Button, LinearProgress, Box, Typography } from '@mui/material';

function TrainingPage() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('Initializing...');
  const [isComplete, setIsComplete] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({});
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    // Get session ID and config from localStorage
    const storedSessionId = localStorage.getItem('trainingSessionId');
    const storedConfig = localStorage.getItem('trainingConfig');
    
    if (!storedSessionId) {
      // If no session ID, redirect back to select page
      navigate('/select');
      return;
    }

    setSessionId(storedSessionId);
    if (storedConfig) {
      try {
        setTrainingConfig(JSON.parse(storedConfig));
      } catch (e) {
        console.error('Failed to parse training config:', e);
      }
    }
  }, [navigate]);

  useEffect(() => {
    if (!sessionId) return;

    const pollProgress = async () => {
      try {
        const response = await fetch(`/api/training/progress?session_id=${sessionId}`);
        const data = await response.json();
        
        if (response.ok) {
          setProgress(data.progress);
          setCurrentStep(data.current_step);
          
          // Check for completion
          if (data.status === 'completed') {
            setIsComplete(true);
            
            // Keep session ID for results page to fetch data
            // No need to pre-fetch results here
            
            setTimeout(() => {
              navigate("/custom_result");
            }, 2000); // 2 second delay before redirect
          }
        } else {
          console.error('Failed to get progress:', data.error);
          setCurrentStep('Error occurred during training');
        }
      } catch (error) {
        console.error('Error polling progress:', error);
        setCurrentStep('Connection error');
      }
    };

    // Poll every 2 seconds
    const interval = setInterval(pollProgress, 2000);
    // Initial poll
    pollProgress();

    return () => clearInterval(interval);
  }, [sessionId, navigate]);

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img
        className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover"
        alt="Space background"
        src="/background.svg"
      />

      <Navbar />

      <main className="relative z-10 px-[3.375rem] pt-32 max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <h1 className="font-bold text-white text-5xl mb-6">
            Training in Progress
          </h1>
          <p className="text-white text-xl mb-8">
            Your machine learning model is being trained on the selected dataset.
          </p>
        </div>

        {/* 訓練狀態卡片 */}
        <div className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 backdrop-blur-sm border border-gray-600/30 rounded-2xl p-8 shadow-2xl mb-8">
          <div className="text-center mb-8">
            <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-2xl flex items-center justify-center backdrop-blur-sm border border-blue-500/30">
              <TrendingUpIcon className="text-4xl text-blue-400" />
            </div>
            <Typography variant="h5" className="text-white font-semibold mb-2">
              {currentStep}
            </Typography>
            <Typography variant="body1" className="text-gray-300">
              {Math.round(progress)}% Complete
            </Typography>
          </div>

          {/* 進度條 */}
          <Box sx={{ width: '100%', mb: 4 }}>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 12,
                borderRadius: 6,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 6,
                  background: 'linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%)',
                },
              }}
            />
          </Box>

          {/* 訓練信息 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="p-4 bg-gray-700/30 rounded-xl backdrop-blur-sm border border-gray-600/30">
              <Typography variant="h6" className="text-blue-400 font-semibold mb-1">
                Dataset
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                {trainingConfig.dataset_source === 'nasa' 
                  ? `NASA ${trainingConfig.dataset_name?.toUpperCase() || 'Kepler'}`
                  : 'Custom Upload'}
              </Typography>
            </div>
            <div className="p-4 bg-gray-700/30 rounded-xl backdrop-blur-sm border border-gray-600/30">
              <Typography variant="h6" className="text-green-400 font-semibold mb-1">
                Model
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                {trainingConfig.model_type === 'rf' ? 'Random Forest' :
                 trainingConfig.model_type === 'xgb' ? 'XGBoost' :
                 trainingConfig.model_type === 'nn' ? 'Neural Network' :
                 'Random Forest'}
              </Typography>
            </div>
            <div className="p-4 bg-gray-700/30 rounded-xl backdrop-blur-sm border border-gray-600/30">
              <Typography variant="h6" className="text-purple-400 font-semibold mb-1">
                Status
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                {isComplete ? 'Complete!' : 'Training...'}
              </Typography>
            </div>
          </div>
        </div>

        {/* 底部提示 */}
        {isComplete && (
          <div className="text-center">
            <div className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-400/30 rounded-xl backdrop-blur-sm">
              <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center mr-3">
                <span className="text-white text-sm">✓</span>
              </div>
              <Typography variant="body1" className="text-green-200 font-medium">
                Training completed! Redirecting to results page...
              </Typography>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default TrainingPage;
