import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
function AboutPage() {
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
          About Us
        </h1>

        <div className="max-w-4xl text-white text-lg leading-relaxed mb-10">
          <p className="mb-6">
          Our project, Exoplanet Playground, empowers users to explore the unknown through the power of machine learning. The website features two main sections: one introduces exoplanets and machine learning fundamentals, helping users without prior background understand the basics. 
          </p>
          <p className="mb-6">
            Learn about how scientists discover exoplanets, the data they collect, and how machine learning can help classify these distant worlds.
          </p>
        </div>

      </main>
    </div>
  );
}

export default AboutPage;
