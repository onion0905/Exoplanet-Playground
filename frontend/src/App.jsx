import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ThemeProvider from "./components/ThemeProvider";
import HomePage from "./pages/home/HomePage";
import AboutPage from "./pages/home/AboutPage";
import ContactPage from "./pages/home/ContactPage";
import LearnExoPage from "./pages/learn/LearnExoPage";
import LearnMLPage from "./pages/learn/LearnMLPage";
import CustomPage from "./pages/select/CustomPage";
import CustomProgressPage from "./pages/progress/CustomProgressPage";
import CustomResultPage from "./pages/result/CustomResultPage";
import PretrainedPage from "./pages/select/PretrainedPage";
import PretrainedProgressPage from "./pages/progress/PretrainedProgressPage";
import PretrainedResultPage from "./pages/result/PretrainedResultPage";

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/learn_exo" element={<LearnExoPage />} />
          <Route path="/learn_ml" element={<LearnMLPage />} />
          <Route path="/custom" element={<CustomPage />} />
          <Route path="/custom/progress" element={<CustomProgressPage />} />
          <Route path="/custom/result" element={<CustomResultPage />} />
          <Route path="/pretrained" element={<PretrainedPage />} />
          <Route path="/pretrained/progress" element={<PretrainedProgressPage />} />
          <Route path="/pretrained/result" element={<PretrainedResultPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
