# NewsNudge

AI-powered fake news detection and personalized news recommendations.

## Overview
NewsNudge is a full-stack app that lets users:
- Analyze text to detect whether it is likely fake or real.
- Get credible, personalized news recommendations.
- View model performance metrics.

The frontend is a React (CRA + CRACO) app styled with Tailwind and Radix UI. The backend is a FastAPI service that loads a pre-trained scikit-learn model and fetches news from The Guardian API.

## Tech Stack
- Frontend: React, CRACO, Tailwind CSS, Radix UI, Axios, React Router
- Backend: FastAPI, scikit-learn, NLTK, MongoDB (Motor), Uvicorn

## Project Structure
- backend/        FastAPI server, ML model, training script
- frontend/       React app
- render.yaml     Render deployment config

## Environment Variables

### Backend (FastAPI)
Create backend/.env with:
- MONGO_URL=your_mongo_connection_string
- DB_NAME=your_database_name
- CORS_ORIGINS=http://localhost:3000

### Frontend (React)
Create or edit frontend/.env with:
- REACT_APP_BACKEND_URL=http://localhost:8000

## Local Development

### 1) Backend
From the repo root:

1. Create and activate a virtual environment.
2. Install dependencies:
	- pip install -r backend/requirements.txt
3. Run the server:
	- uvicorn server:app --reload --host 0.0.0.0 --port 8000 --app-dir backend

API will be available at http://localhost:8000.

### 2) Frontend
From the frontend folder:

1. Install dependencies:
	- npm install --legacy-peer-deps
2. Start the dev server:
	- npm start

App will be available at http://localhost:3000.

## API Endpoints
Base path: /api

- GET /api/                Health message
- POST /api/detect-fake    Fake news detection
- POST /api/recommend-news News recommendations
- GET /api/news/{category} Latest news by category
- GET /api/metrics         Model metrics
- GET /api/history         Query history (MongoDB)

## Model Training (Optional)
To retrain the model and regenerate metrics:
- python backend/train_model.py

This will regenerate the model files in backend/ml_models and update model_metrics.json.

## Deployment
Render configuration is provided in render.yaml for a Dockerized backend and a static frontend build.
