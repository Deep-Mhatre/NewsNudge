import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib
import json
from pathlib import Path

# Download required NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

# Create sample fake news dataset
def create_sample_dataset():
    """Create a sample fake news dataset for training"""
    
    # Real news samples
    real_news = [
        "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. The research team published their findings in Nature magazine.",
        "The Federal Reserve announced today that interest rates will remain unchanged at 5.5 percent, citing economic stability.",
        "A new study from Harvard Medical School shows that regular exercise can reduce the risk of heart disease by up to 30 percent.",
        "The United Nations climate summit concluded with 195 countries agreeing to reduce carbon emissions by 2030.",
        "NASA's James Webb telescope captured stunning images of a distant galaxy formed 13 billion years ago.",
        "The Supreme Court will hear arguments on a landmark case regarding digital privacy rights next month.",
        "Major technology companies announced plans to invest $50 billion in renewable energy infrastructure over the next decade.",
        "Researchers at MIT developed a new material that could make batteries last three times longer than current technology.",
        "The World Health Organization reported a significant decline in malaria cases in Africa due to improved prevention methods.",
        "Economic data shows unemployment rates have dropped to 3.8 percent, the lowest in two years.",
        "A peer-reviewed study published in Science journal reveals new insights into Alzheimer's disease progression.",
        "The European Union reached a trade agreement with Asian nations worth $200 billion annually.",
        "Climate scientists warn that Arctic ice is melting faster than previously predicted based on satellite data.",
        "A major archaeological discovery in Egypt reveals previously unknown details about ancient civilization.",
        "The Nobel Prize in Physics was awarded to researchers for their work on quantum computing.",
        "New legislation aims to strengthen cybersecurity measures for critical infrastructure nationwide.",
        "Medical researchers announced progress in developing a universal flu vaccine that could work for multiple years.",
        "The International Space Station completed its 100,000th orbit around Earth after 23 years in operation.",
        "Economic forecasts predict moderate GDP growth of 2.5 percent for the upcoming fiscal year.",
        "A comprehensive study shows that renewable energy now accounts for 30 percent of global electricity production.",
    ] * 25  # Multiply to get 500 samples
    
    # Fake news samples
    fake_news = [
        "BREAKING: Government secretly testing mind control technology on citizens through 5G towers and vaccines!",
        "You won't believe what this celebrity said! Doctors hate them for revealing this one weird trick!",
        "Shocking discovery: Ancient aliens built the pyramids and the government has been hiding the evidence!",
        "Scientists confirm that eating chocolate cake for breakfast will make you lose 50 pounds in one week!",
        "URGENT: New law will ban all pets starting next month. Share this before they delete it!",
        "Billionaire reveals secret: This simple method will make you rich overnight! Banks don't want you to know!",
        "Breaking news: Famous actor claims to have discovered the fountain of youth in their backyard!",
        "Unbelievable: Man survives 40 days without food or water, doctors are baffled by this miracle!",
        "Government insider leaks documents proving that birds are actually surveillance drones!",
        "Revolutionary discovery: Scientists prove that the Earth is actually flat and NASA has been lying!",
        "Miracle cure discovered! This common household item cures cancer, diabetes, and all diseases instantly!",
        "Shocking truth revealed: Moon landing was faked in a Hollywood studio, whistleblower exposes conspiracy!",
        "Celebrity predicts the end of the world next Tuesday based on ancient prophecy! Prepare now!",
        "Incredible breakthrough: Drink this juice and never age again! Big pharma is trying to suppress it!",
        "Government plans to implant tracking chips in everyone next month! Wake up people!",
        "Amazing discovery: Bermuda Triangle mystery finally solved by local fisherman! You won't believe this!",
        "Scientists baffled: Woman gives birth to alien baby with superpowers! Photos leaked online!",
        "Breaking: Popular food item contains secret ingredient that controls your thoughts! Share immediately!",
        "Insider reveals: All politicians are actually reptilian aliens in disguise! Evidence finally surfaces!",
        "Miracle weight loss: Lose 100 pounds in 3 days without diet or exercise! Doctors stunned!",
    ] * 25  # Multiply to get 500 samples
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': real_news + fake_news,
        'label': [0] * len(real_news) + [1] * len(fake_news)  # 0 = real, 1 = fake
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text

def train_fake_news_detector():
    """Train the fake news detection model"""
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Save raw dataset
    df.to_csv('/app/backend/data/fake_news_dataset.csv', index=False)
    print(f"Dataset created with {len(df)} samples")
    print(f"Real news: {len(df[df['label']==0])}, Fake news: {len(df[df['label']==1])}")
    
    # Preprocess text
    print("\nPreprocessing text data...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create TF-IDF vectorizer
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"\nF1 Score: {f1:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"True Negatives (Real as Real): {cm[0][0]}")
    print(f"False Positives (Real as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake as Real): {cm[1][0]}")
    print(f"True Positives (Fake as Fake): {cm[1][1]}")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save model and vectorizer
    print("\n" + "="*50)
    print("Saving model and vectorizer...")
    joblib.dump(model, '/app/backend/ml_models/fake_news_model.pkl')
    joblib.dump(vectorizer, '/app/backend/ml_models/tfidf_vectorizer.pkl')
    
    # Save metrics
    metrics = {
        'f1_score': float(f1),
        'roc_auc_score': float(roc_auc),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=['Real', 'Fake'], output_dict=True)
    }
    
    with open('/app/backend/ml_models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model saved to: /app/backend/ml_models/fake_news_model.pkl")
    print("Vectorizer saved to: /app/backend/ml_models/tfidf_vectorizer.pkl")
    print("Metrics saved to: /app/backend/ml_models/model_metrics.json")
    print("="*50)
    
    return model, vectorizer, metrics

if __name__ == "__main__":
    train_fake_news_detector()
