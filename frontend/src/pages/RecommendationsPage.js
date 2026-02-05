import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Sparkles, Search, ExternalLink, Loader2, TrendingUp } from "lucide-react";
import axios from "axios";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const RecommendationsPage = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.error("Please enter a topic to search");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/recommend-news`, { 
        query,
        limit: 8
      });
      setRecommendations(response.data);
      if (response.data.count === 0) {
        toast.info("No articles found for this topic. Try a different query.");
      } else {
        toast.success(`Found ${response.data.count} relevant articles!`);
      }
    } catch (error) {
      console.error("Error getting recommendations:", error);
      toast.error("Failed to get recommendations. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const quickTopics = [
    { label: "Climate Change" },
    { label: "Technology" },
    { label: "Health" },
    { label: "Sports" },
    { label: "Politics" },
    { label: "Science" }
  ];

  const handleQuickTopic = (topic) => {
    setQuery(topic);
    setRecommendations(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 via-white to-blue-50">
      <div className="max-w-6xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-8">
          <Button
            data-testid="back-btn"
            onClick={() => navigate("/")}
            variant="ghost"
            className="mb-4 hover:bg-teal-100"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
          
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-teal-100 p-3 rounded-lg">
              <Sparkles className="w-8 h-8 text-teal-600" />
            </div>
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-gray-800">News Recommendations</h1>
              <p className="text-gray-600">Discover credible news tailored to your interests</p>
            </div>
          </div>
        </div>

        {/* Search Section */}
        <Card className="mb-6 shadow-lg border-teal-200">
          <CardHeader>
            <CardTitle className="text-xl">What topics interest you?</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-3 mb-4">
              <Input
                data-testid="query-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="e.g., artificial intelligence, renewable energy, space exploration..."
                className="text-base"
              />
              <Button
                data-testid="search-btn"
                onClick={handleSearch}
                disabled={loading || !query.trim()}
                className="bg-teal-600 hover:bg-teal-700 text-white px-8 rounded-xl shadow-lg hover:shadow-xl transition-all"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <Search className="w-5 h-5 mr-2" />
                    Search
                  </>
                )}
              </Button>
            </div>
            
            <div className="flex flex-wrap gap-2">
              <span className="text-sm text-gray-600 mr-2">Quick topics:</span>
              {quickTopics.map((topic, index) => (
                <Button
                  key={index}
                  data-testid={`quick-topic-${topic.label.toLowerCase().replace(' ', '-')}`}
                  onClick={() => handleQuickTopic(topic.label)}
                  variant="outline"
                  size="sm"
                  className="border-teal-300 text-teal-700 hover:bg-teal-50"
                >
                  {topic.label}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Results Section */}
        {recommendations && (
          <div>
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">
                Results for "{recommendations.query}"
              </h2>
              <p className="text-gray-600">
                Found {recommendations.count} credible article{recommendations.count !== 1 ? 's' : ''}
              </p>
            </div>

            {recommendations.articles.length > 0 ? (
              <div className="grid md:grid-cols-2 gap-6">
                {recommendations.articles.map((article, index) => (
                  <Card
                    key={index}
                    data-testid={`article-${index}`}
                    className="shadow-lg hover:shadow-xl transition-all border-teal-200 overflow-hidden group"
                  >
                    {article.image_url && (
                      <div className="w-full h-48 overflow-hidden bg-gray-100">
                        <img
                          src={article.image_url}
                          alt={article.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                    )}
                    <CardContent className="p-6">
                      <div className="flex items-center gap-2 mb-3">
                        <Badge className="bg-teal-100 text-teal-700 hover:bg-teal-200">
                          {article.source}
                        </Badge>
                        {article.credibility_score && (
                          <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-200">
                            <TrendingUp className="w-3 h-3 mr-1" />
                            {(article.credibility_score * 100).toFixed(0)}% credible
                          </Badge>
                        )}
                      </div>
                      
                      <h3 className="text-lg font-bold mb-2 text-gray-800 line-clamp-2">
                        {article.title}
                      </h3>
                      
                      {article.description && (
                        <p className="text-sm text-gray-600 mb-4 line-clamp-3">
                          {article.description.replace(/<[^>]*>/g, '')}
                        </p>
                      )}
                      
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500">
                          {new Date(article.published_at).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                          })}
                        </span>
                        
                        <Button
                          data-testid={`read-article-${index}`}
                          onClick={() => window.open(article.url, '_blank')}
                          variant="ghost"
                          size="sm"
                          className="text-teal-600 hover:text-teal-700 hover:bg-teal-50"
                        >
                          Read Article
                          <ExternalLink className="w-4 h-4 ml-2" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="shadow-lg border-gray-200">
                <CardContent className="p-12 text-center">
                  <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-gray-700 mb-2">No articles found</h3>
                  <p className="text-gray-600">Try searching for a different topic or use one of the quick topics above.</p>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default RecommendationsPage;


