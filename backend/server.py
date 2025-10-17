from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import joblib
import re
from nltk.corpus import stopwords
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Load ML model and vectorizer
model = joblib.load('/app/backend/ml_models/fake_news_model.pkl')
tfidf_vectorizer = joblib.load('/app/backend/ml_models/tfidf_vectorizer.pkl')

# Load model metrics
with open('/app/backend/ml_models/model_metrics.json', 'r') as f:
    model_metrics = json.load(f)

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class FakeNewsRequest(BaseModel):
    text: str

class FakeNewsResponse(BaseModel):
    is_fake: bool
    confidence: float
    prediction: str
    text_preview: str

class RecommendationRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class NewsArticle(BaseModel):
    title: str
    description: Optional[str]
    url: str
    source: str
    published_at: str
    image_url: Optional[str]
    credibility_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    query: str
    articles: List[NewsArticle]
    count: int

class ModelMetrics(BaseModel):
    f1_score: float
    roc_auc_score: float
    accuracy: float
    confusion_matrix: List[List[int]]

class QueryHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str
    query_type: str  # 'fake_detection' or 'recommendation'
    result: dict
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper functions
def preprocess_text(text: str) -> str:
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text

async def save_query_history(query_text: str, query_type: str, result: dict):
    """Save query to MongoDB history"""
    try:
        history = QueryHistory(
            query_text=query_text,
            query_type=query_type,
            result=result
        )
        doc = history.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.query_history.insert_one(doc)
    except Exception as e:
        logger.error(f"Error saving query history: {e}")

# API Routes
@api_router.get("/")
async def root():
    return {"message": "NEWSNUDGE API - Fake News Detector & Personalized News Assistant"}

@api_router.post("/detect-fake", response_model=FakeNewsResponse)
async def detect_fake_news(request: FakeNewsRequest):
    """Detect if a news article is fake or real"""
    try:
        # Preprocess text
        cleaned_text = preprocess_text(request.text)
        
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text is too short or invalid")
        
        # Transform text using TF-IDF
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        is_fake = bool(prediction == 1)
        confidence = float(probability[1] if is_fake else probability[0])
        
        result = {
            "is_fake": is_fake,
            "confidence": confidence,
            "prediction": "Fake News" if is_fake else "Real News",
            "text_preview": request.text[:150] + "..." if len(request.text) > 150 else request.text
        }
        
        # Save to history
        await save_query_history(request.text, "fake_detection", result)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in fake news detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/news/{category}")
async def get_news_by_category(category: str, limit: int = 10):
    """Get latest news articles by category from The Guardian API"""
    try:
        # The Guardian API endpoint (open API, no key needed for basic access)
        url = f"https://content.guardianapis.com/search"
        
        params = {
            "section": category,
            "page-size": limit,
            "show-fields": "thumbnail,trailText",
            "order-by": "newest"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = []
            
            for item in data.get('response', {}).get('results', []):
                fields = item.get('fields', {})
                articles.append({
                    "title": item.get('webTitle', ''),
                    "description": fields.get('trailText', ''),
                    "url": item.get('webUrl', ''),
                    "source": "The Guardian",
                    "published_at": item.get('webPublicationDate', ''),
                    "image_url": fields.get('thumbnail', ''),
                    "credibility_score": 0.95  # The Guardian is a credible source
                })
            
            return {"category": category, "articles": articles, "count": len(articles)}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch news")
    
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/recommend-news", response_model=RecommendationResponse)
async def recommend_news(request: RecommendationRequest):
    """Get personalized news recommendations based on user query"""
    try:
        # Fetch news articles from The Guardian
        url = "https://content.guardianapis.com/search"
        params = {
            "q": request.query,
            "page-size": 20,
            "show-fields": "thumbnail,trailText,bodyText",
            "order-by": "relevance"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch news articles")
        
        data = response.json()
        results = data.get('response', {}).get('results', [])
        
        if not results:
            return RecommendationResponse(query=request.query, articles=[], count=0)
        
        # Extract article texts and create TF-IDF matrix
        article_texts = []
        article_data = []
        
        for item in results:
            fields = item.get('fields', {})
            text = fields.get('bodyText', '') or fields.get('trailText', '') or item.get('webTitle', '')
            article_texts.append(text)
            article_data.append({
                "title": item.get('webTitle', ''),
                "description": fields.get('trailText', ''),
                "url": item.get('webUrl', ''),
                "source": "The Guardian",
                "published_at": item.get('webPublicationDate', ''),
                "image_url": fields.get('thumbnail', '')
            })
        
        # Create new TF-IDF vectorizer for similarity calculation
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Fit on article texts and query
        all_texts = [request.query] + article_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between query and articles
        query_vector = tfidf_matrix[0:1]
        article_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(query_vector, article_vectors)[0]
        
        # Sort articles by similarity
        sorted_indices = np.argsort(similarities)[::-1][:request.limit]
        
        # Create recommended articles with credibility scores
        recommended_articles = []
        for idx in sorted_indices:
            if similarities[idx] > 0.1:  # Only include relevant articles
                article = article_data[idx]
                article['credibility_score'] = 0.95  # The Guardian is credible
                recommended_articles.append(NewsArticle(**article))
        
        result = {
            "query": request.query,
            "articles": [article.model_dump() for article in recommended_articles],
            "count": len(recommended_articles)
        }
        
        # Save to history
        await save_query_history(request.query, "recommendation", result)
        
        return RecommendationResponse(
            query=request.query,
            articles=recommended_articles,
            count=len(recommended_articles)
        )
    
    except Exception as e:
        logger.error(f"Error in news recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics"""
    return ModelMetrics(**model_metrics)

@api_router.get("/history")
async def get_query_history(limit: int = 20):
    """Get recent query history"""
    try:
        history = await db.query_history.find({}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
