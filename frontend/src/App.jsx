import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ThemeProvider from "./components/ThemeProvider";
import HomePage from "./pages/home/HomePage";
import LearnPage from "./pages/learn/LearnPage";
import SelectPage from "./pages/select/SelectPage";
import TrainingPage from "./pages/training/TrainingPage";
import PredictPage from "./pages/predict/PredictPage";
import SimpleResultPage from "./pages/result/SimpleResultPage";

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/learn" element={<LearnPage />} />
          <Route path="/select" element={<SelectPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/custom_result" element={<SimpleResultPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
