import { useState } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import DetectorPage from "./pages/DetectorPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import { Toaster } from "./components/ui/sonner";

function App() {
  return (
    <div className="App">
      <Toaster position="top-right" />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/detector" element={<DetectorPage />} />
          <Route path="/recommendations" element={<RecommendationsPage />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
