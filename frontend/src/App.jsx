import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ThemeProvider from "./components/ThemeProvider";
import HomePage from "./pages/home/HomePage";
import LearnExoPage from "./pages/learn/LearnExoPage";
import LearnMLPage from "./pages/learn/LearnMLPage";
import SelectPage from "./pages/select/SelectPage";
import PretrainedPage from "./pages/select/PretrainedPage";
import TrainingPage from "./pages/training/TrainingPage";
import PredictPage from "./pages/predict(dropped)/PredictPage";
import CustomResultPage from "./pages/result/CustomResultPage";
import PretrainedResultPage from "./pages/result/PretrainedResultPage";

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/learn_exo" element={<LearnExoPage />} />
          <Route path="/learn_ml" element={<LearnMLPage />} />
          <Route path="/select" element={<SelectPage />} />
          <Route path="/pretrained" element={<PretrainedPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/custom_result" element={<CustomResultPage />} />
          <Route path="/pretrained_result" element={<PretrainedResultPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
