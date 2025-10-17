import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Shield, Sparkles, TrendingUp, CheckCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { useEffect, useState } from "react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const HomePage = () => {
  const navigate = useNavigate();
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API}/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error("Error fetching metrics:", error);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-teal-50"></div>
        
        <div className="relative max-w-7xl mx-auto px-6 py-20">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full border border-blue-200 mb-6">
              <Shield className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-700">Powered by Advanced ML</span>
            </div>
            
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-700 via-teal-600 to-blue-700 bg-clip-text text-transparent">
              NEWSNUDGE
            </h1>
            
            <p className="text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto mb-8">
              Combat misinformation with AI-powered fake news detection and get personalized credible news recommendations
            </p>
            
            <div className="flex flex-wrap justify-center gap-4">
              <Button
                data-testid="detector-btn"
                onClick={() => navigate("/detector")}
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-6 text-lg rounded-xl shadow-lg hover:shadow-xl transition-all"
              >
                <Shield className="w-5 h-5 mr-2" />
                Detect Fake News
              </Button>
              
              <Button
                data-testid="recommendations-btn"
                onClick={() => navigate("/recommendations")}
                size="lg"
                variant="outline"
                className="border-2 border-blue-600 text-blue-700 hover:bg-blue-50 px-8 py-6 text-lg rounded-xl shadow-lg hover:shadow-xl transition-all"
              >
                <Sparkles className="w-5 h-5 mr-2" />
                Get Recommendations
              </Button>
            </div>
          </div>

          {/* Model Performance Section */}
          {metrics && (
            <div className="mb-16">
              <h2 className="text-2xl font-bold text-center mb-8 text-gray-800">Model Performance</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                <Card className="bg-white/80 backdrop-blur-sm border-blue-200 shadow-lg hover:shadow-xl transition-all">
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <TrendingUp className="w-12 h-12 text-blue-600 mx-auto mb-3" />
                      <p className="text-sm text-gray-600 mb-2">F1 Score</p>
                      <p className="text-3xl font-bold text-blue-700">{(metrics.f1_score * 100).toFixed(1)}%</p>
                    </div>
                  </CardContent>
                </Card>
                
                <Card className="bg-white/80 backdrop-blur-sm border-teal-200 shadow-lg hover:shadow-xl transition-all">
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <CheckCircle className="w-12 h-12 text-teal-600 mx-auto mb-3" />
                      <p className="text-sm text-gray-600 mb-2">Accuracy</p>
                      <p className="text-3xl font-bold text-teal-700">{(metrics.accuracy * 100).toFixed(1)}%</p>
                    </div>
                  </CardContent>
                </Card>
                
                <Card className="bg-white/80 backdrop-blur-sm border-blue-200 shadow-lg hover:shadow-xl transition-all">
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <Sparkles className="w-12 h-12 text-blue-600 mx-auto mb-3" />
                      <p className="text-sm text-gray-600 mb-2">ROC-AUC Score</p>
                      <p className="text-3xl font-bold text-blue-700">{(metrics.roc_auc_score * 100).toFixed(1)}%</p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {/* Features Section */}
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            <Card className="bg-white/80 backdrop-blur-sm border-blue-200 shadow-lg hover:shadow-xl transition-all">
              <CardContent className="p-8">
                <div className="flex items-start gap-4">
                  <div className="bg-blue-100 p-3 rounded-lg">
                    <Shield className="w-8 h-8 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2 text-gray-800">Fake News Detection</h3>
                    <p className="text-gray-600">
                      Advanced ML model using TF-IDF and Logistic Regression to identify fake news with high accuracy. Get instant credibility scores.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-white/80 backdrop-blur-sm border-teal-200 shadow-lg hover:shadow-xl transition-all">
              <CardContent className="p-8">
                <div className="flex items-start gap-4">
                  <div className="bg-teal-100 p-3 rounded-lg">
                    <Sparkles className="w-8 h-8 text-teal-600" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2 text-gray-800">Personalized Recommendations</h3>
                    <p className="text-gray-600">
                      Discover credible news tailored to your interests using cosine similarity matching. Stay informed with trusted sources.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
