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

  // 模擬訓練步驟
  const trainingSteps = [
    { step: 'Initializing...', duration: 1000 },
    { step: 'Loading dataset...', duration: 2000 },
    { step: 'Preprocessing data...', duration: 3000 },
    { step: 'Training model...', duration: 4000 },
    { step: 'Validating results...', duration: 2000 },
    { step: 'Saving model...', duration: 1000 },
    { step: 'Training complete!', duration: 500 }
  ];

  useEffect(() => {
    let currentStepIndex = 0;
    let totalDuration = 0;
    
    // 計算總時長
    const totalTime = trainingSteps.reduce((acc, step) => acc + step.duration, 0);
    
    const interval = setInterval(() => {
      if (currentStepIndex < trainingSteps.length) {
        const step = trainingSteps[currentStepIndex];
        setCurrentStep(step.step);
        
        // 更新進度
        const stepProgress = ((currentStepIndex + 1) / trainingSteps.length) * 100;
        setProgress(stepProgress);
        
        // 如果是最後一步，標記為完成
        if (currentStepIndex === trainingSteps.length - 1) {
          setIsComplete(true);
          setTimeout(() => {
            navigate("/custom_result");
          }, 2000); // 2秒後跳轉到 result 頁面
        }
        
        currentStepIndex++;
      }
    }, 1000);

    return () => clearInterval(interval);
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
                NASA Kepler
              </Typography>
            </div>
            <div className="p-4 bg-gray-700/30 rounded-xl backdrop-blur-sm border border-gray-600/30">
              <Typography variant="h6" className="text-green-400 font-semibold mb-1">
                Model
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                Random Forest
              </Typography>
            </div>
            <div className="p-4 bg-gray-700/30 rounded-xl backdrop-blur-sm border border-gray-600/30">
              <Typography variant="h6" className="text-purple-400 font-semibold mb-1">
                ETA
              </Typography>
              <Typography variant="body2" className="text-gray-300">
                {isComplete ? 'Complete!' : '~2 min'}
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
