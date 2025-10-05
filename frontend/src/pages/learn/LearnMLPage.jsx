import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Button } from '@mui/material';

function LearnPage() {
  const navigate = useNavigate();

  return (
    <div className="relative w-full min-h-screen bg-[#14171e] overflow-hidden">
      <img
        className="absolute top-20 left-0 w-full h-[calc(100vh-5rem)] object-cover"
        alt="Space background"
        src="/background.svg"
      />

      <Navbar />

      <main className="relative z-10 px-[3.375rem] pt-32">
        <h1 className="font-bold text-white text-5xl mb-10">
          Learn About Exoplanets & Machine Learning
        </h1>

        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
            This page will contain educational content about exoplanets and machine learning concepts.
          </p>
          <p className="mb-6">
            Learn about how scientists discover exoplanets, the data they collect, and how machine learning can help classify these distant worlds.
          </p>
        </div>

        <div className="flex gap-[1.75rem]">
          <Button
            variant="outlined"
            size="large"
            className="h-14 w-60 text-xl flex items-center gap-3"
            onClick={() => navigate("/select")}
            sx={{
              color: '#f5eff7',
              borderColor: '#f5eff7',
              '&:hover': {
                borderColor: '#f5eff7',
                backgroundColor: 'rgba(245, 239, 247, 0.1)'
              }
            }}
          >
            <PlayArrowIcon className="text-xl" />
            Start Training
          </Button>
        </div>
      </main>
    </div>
  );
}

export default LearnPage;
