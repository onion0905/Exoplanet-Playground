import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import { Button, LinearProgress, Box, Typography, Alert } from '@mui/material';
import { pretrainedPrediction } from '../../lib/api';

function PretrainedProgressPage() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('Initializing...');
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState({ dataset: 'Loading...', model: 'Pretrained Model' });

  useEffect(() => {
    const sessionId = sessionStorage.getItem('pretrained_session_id');
    
    if (!sessionId) {
      setError('No prediction session found. Please start prediction first.');
      setTimeout(() => navigate("/pretrained"), 2000);
      return;
    }

    let pollInterval;
    
    const pollProgress = async () => {
      try {
        const data = await pretrainedPrediction.getProgress(sessionId);
        
        setProgress(data.progress);
        setCurrentStep(data.current_step);
        
        if (data.status === 'completed') {
          setIsComplete(true);
          clearInterval(pollInterval);
          setTimeout(() => {
            navigate("/pretrained/result");
          }, 2000);
        } else if (data.status === 'error') {
          setError('Prediction failed. Please try again.');
          clearInterval(pollInterval);
        }
      } catch (err) {
        console.error('Progress polling error:', err);
        setError(err.message || 'Failed to get prediction progress');
        clearInterval(pollInterval);
      }
    };

    pollProgress();
    pollInterval = setInterval(pollProgress, 1000);

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [navigate]);

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

        {error && (
          <Alert severity="error" sx={{ mb: 4 }}>
            {error}
          </Alert>
        )}

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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
            <div className="p-4 bg-gray-700/30 rounded-xl backdrop-blur-sm border border-gray-600/30">
              <Typography variant="h6" className="text-blue-400 font-semibold mb-1">
                Dataset
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                {datasetInfo.dataset}
              </Typography>
            </div>
            <div className="p-4 bg-gray-700/30 rounded-xl backdrop-blur-sm border border-gray-600/30">
              <Typography variant="h6" className="text-green-400 font-semibold mb-1">
                Model
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                {datasetInfo.model}
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

export default PretrainedProgressPage;
