import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import TravelExploreIcon from '@mui/icons-material/TravelExplore';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Button } from '@mui/material';

const actionButtons = [
  { label: "Learn about Exoplanet", path: "/learn", icon: TravelExploreIcon },
  { label: "Learn about ML", path: "/learn", icon: PsychologyIcon },
  { label: "Launch the model", path: "/select", icon: PlayArrowIcon},
];

function HomePage() {
  const navigate = useNavigate();

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
          Introduction
        </h1>

        <p className="text-white text-xl leading-relaxed mb-10 max-w-4xl">
          Welcome to the Exoplanet AI Playground.
          <br />
          Here, you can explore how scientists search for worlds beyond our
          solar system. Learn about exoplanets, experiment with NASA data or
          your own dataset, and train machine learning models to classify new
          worlds. Adjust parameters, test predictions, and visualize results in
          3D.
        </p>

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
