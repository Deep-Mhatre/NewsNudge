import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ArrowLeft, Shield, AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import axios from "axios";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DetectorPage = () => {
  const navigate = useNavigate();
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleDetect = async () => {
    if (!text.trim()) {
      toast.error("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/detect-fake`, { text });
      setResult(response.data);
      toast.success("Analysis complete!");
    } catch (error) {
      console.error("Error detecting fake news:", error);
      toast.error("Failed to analyze the text. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getSampleText = (type) => {
    const samples = {
      real: "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. The research team from the University of Washington published their findings in the journal Nature, detailing the unique characteristics of the bioluminescent creature found at depths exceeding 3,000 meters.",
      fake: "BREAKING: Government secretly testing mind control technology on citizens through 5G towers! Whistleblower reveals shocking documents proving that vaccines contain microchips. This is the biggest conspiracy of our time! Share before they delete this!"
    };
    setText(samples[type]);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50">
      <div className="max-w-5xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-8">
          <Button
            data-testid="back-btn"
            onClick={() => navigate("/")}
            variant="ghost"
            className="mb-4 hover:bg-blue-100"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
          
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-blue-100 p-3 rounded-lg">
              <Shield className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-gray-800">Fake News Detector</h1>
              <p className="text-gray-600">Analyze any text to check its credibility</p>
            </div>
          </div>
        </div>

        {/* Input Section */}
        <Card className="mb-6 shadow-lg border-blue-200">
          <CardHeader>
            <CardTitle className="text-xl">Enter Text to Analyze</CardTitle>
          </CardHeader>
          <CardContent>
            <Textarea
              data-testid="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste a news article or any text you want to verify..."
              className="min-h-[200px] text-base mb-4"
            />
            
            <div className="flex flex-wrap gap-3 mb-4">
              <Button
                data-testid="sample-real-btn"
                onClick={() => getSampleText('real')}
                variant="outline"
                size="sm"
                className="border-teal-400 text-teal-700 hover:bg-teal-50"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Try Real News Sample
              </Button>
              
              <Button
                data-testid="sample-fake-btn"
                onClick={() => getSampleText('fake')}
                variant="outline"
                size="sm"
                className="border-red-400 text-red-700 hover:bg-red-50"
              >
                <AlertTriangle className="w-4 h-4 mr-2" />
                Try Fake News Sample
              </Button>
            </div>
            
            <Button
              data-testid="analyze-btn"
              onClick={handleDetect}
              disabled={loading || !text.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-6 text-lg rounded-xl shadow-lg hover:shadow-xl transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Shield className="w-5 h-5 mr-2" />
                  Analyze Text
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Result Section */}
        {result && (
          <Card className={`shadow-lg ${
            result.is_fake 
              ? 'border-red-300 bg-red-50/50' 
              : 'border-teal-300 bg-teal-50/50'
          }`}>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                {result.is_fake ? (
                  <>
                    <AlertTriangle className="w-8 h-8 text-red-600" />
                    <span className="text-red-700">Potentially Fake News Detected</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-8 h-8 text-teal-600" />
                    <span className="text-teal-700">Appears to be Real News</span>
                  </>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert className={result.is_fake ? 'border-red-300' : 'border-teal-300'}>
                  <AlertDescription className="text-base">
                    <div className="mb-3">
                      <span className="font-semibold">Prediction: </span>
                      <span data-testid="prediction-result" className={`font-bold ${
                        result.is_fake ? 'text-red-700' : 'text-teal-700'
                      }`}>
                        {result.prediction}
                      </span>
                    </div>
                    <div>
                      <span className="font-semibold">Confidence Score: </span>
                      <span data-testid="confidence-score" className="font-bold">
                        {(result.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                  </AlertDescription>
                </Alert>
                
                <div className="bg-white p-4 rounded-lg border">
                  <p className="text-sm text-gray-600 mb-2 font-semibold">Analyzed Text Preview:</p>
                  <p data-testid="text-preview" className="text-gray-700">{result.text_preview}</p>
                </div>

                <div className="bg-white p-4 rounded-lg border">
                  <p className="text-sm font-semibold text-gray-800 mb-2">
                    {result.is_fake ? '⚠️ Warning Signs:' : '✓ Credibility Indicators:'}
                  </p>
                  <ul className="text-sm text-gray-700 space-y-1 list-disc list-inside">
                    {result.is_fake ? (
                      <>
                        <li>Sensational language and emotional manipulation detected</li>
                        <li>Lacks verifiable sources or scientific evidence</li>
                        <li>Contains typical fake news patterns</li>
                        <li>Recommend cross-checking with trusted sources</li>
                      </>
                    ) : (
                      <>
                        <li>Professional language and balanced tone</li>
                        <li>Contains factual information patterns</li>
                        <li>Follows credible news structure</li>
                        <li>Likely from legitimate news source</li>
                      </>
                    )}
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default DetectorPage;
