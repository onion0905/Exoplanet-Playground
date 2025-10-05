import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import CSSBackgroundVisualization from "../../components/CSSBackgroundVisualization";
function AboutPage() {
  const navigate = useNavigate();

  return (
    <div className="relative w-full min-h-screen">
      <CSSBackgroundVisualization />
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
          The other section provides a hands-on interactive environment, where users can select and train models to detect exoplanets — even using their own datasets.
          </p>
          <p className="mb-6">
          By training and deploying seven different models on NASA’s exoplanet datasets (KOI, TOI, and K2), we’ve created an engaging platform where users can test, compare, and learn from real data.
          </p>
          <p className="mb-6">
          We hope this platform inspires more people to discover the beauty of exoplanets and to develop a lasting interest in aerospace — which is our ultimate goal! 
          </p>
        </div>

      </main>
    </div>
  );
}

export default AboutPage;
