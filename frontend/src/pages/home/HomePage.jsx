import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import TravelExploreIcon from '@mui/icons-material/TravelExplore';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Button, CircularProgress, Alert } from '@mui/material';
import apiService from '../../lib/api';

const actionButtons = [
  { label: "Learn about Exoplanet", path: "/learn", icon: TravelExploreIcon },
  { label: "Learn about ML", path: "/learn", icon: PsychologyIcon },
  { label: "Launch the model", path: "/select", icon: PlayArrowIcon},
];

function HomePage() {
  const navigate = useNavigate();
  const [homeData, setHomeData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHomeData = async () => {
      try {
        setLoading(true);
        const data = await apiService.getHomeData();
        setHomeData(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch home data:', err);
        setError('Failed to connect to backend. Make sure the Flask server is running on port 5000.');
      } finally {
        setLoading(false);
      }
    };

    fetchHomeData();
  }, []);

  if (loading) {
    return (
      <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden flex items-center justify-center">
        <CircularProgress size={60} sx={{ color: '#60a5fa' }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
        <Navbar />
        <main className="relative z-10 px-[3.375rem] pt-32 mx-10">
          <Alert 
            severity="error" 
            sx={{ 
              mb: 4,
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              color: 'white',
              '& .MuiAlert-icon': { color: '#ef4444' }
            }}
          >
            {error}
          </Alert>
          <h1 className="font-bold text-white text-5xl mb-10">
            Connection Error
          </h1>
          <p className="text-white text-xl leading-relaxed mb-10 max-w-4xl">
            Unable to connect to the backend server. Please make sure:
            <br />
            1. The Flask backend is running on http://localhost:5000
            <br />
            2. Run: python run.py in the project root
          </p>
        </main>
      </div>
    );
  }

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img
        className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover"
        alt="Space background"
        src="/background.svg"
      />

      <Navbar />

      <main className="relative z-10 px-[3.375rem] pt-32 mx-10">
        <h1 className="font-bold text-white text-5xl mb-10">
          {homeData?.title || 'Exoplanet Discovery Playground'}
        </h1>

        <p className="text-white text-xl leading-relaxed mb-10 max-w-4xl">
          {homeData?.description || 'Train your own machine learning models to identify exoplanets using NASA data'}
          <br />
          {homeData?.message || 'Welcome to the Exoplanet Machine Learning Platform!'}
        </p>

        {homeData?.features && (
          <div className="mb-8">
            <h2 className="text-white text-2xl font-semibold mb-4">Features:</h2>
            <ul className="text-white text-lg space-y-2 max-w-4xl">
              {homeData.features.map((feature, index) => (
                <li key={index} className="flex items-start">
                  <span className="text-blue-400 mr-2">•</span>
                  {feature}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex justify-center gap-10">

          <Button
            variant="outlined"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            onClick={() => navigate("/learn")}
            sx={{
              color: '#f5eff7',
              borderColor: '#f5eff7',
              '&:hover': {
                borderColor: '#f5eff7',
                backgroundColor: 'rgba(245, 239, 247, 0.1)'
              }
            }}
          >
            <TravelExploreIcon className="text-xl" />
            Learn about Exoplanet
          </Button>

          <Button
            variant="outlined"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            onClick={() => navigate("/learn")}
            sx={{
              color: '#f5eff7',
              borderColor: '#f5eff7',
              '&:hover': {
                borderColor: '#f5eff7',
                backgroundColor: 'rgba(245, 239, 247, 0.1)'
              }
            }}
          >
            <PsychologyIcon className="text-xl" />
            Learn about ML
          </Button>

          <Button
            variant="contained"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            onClick={() => navigate("/select")}
            sx={{
              backgroundColor: '#2563eb',
              '&:hover': {
                backgroundColor: '#1d4ed8'
              },
              color: 'white'
            }}
          >
            <PlayArrowIcon className="text-xl" />
            Launch the model
          </Button>

        </div>
      </main>
    </div>
  );
}

export default HomePage;
