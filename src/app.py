from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel # Added for better request body validation

# Define the input model for the prediction endpoint
class ReviewInput(BaseModel):
    review: str

# 1. Initialize the FastAPI app instance
app = FastAPI(
    title="Sentiment Analysis API",
    description="Backend for classifying text reviews as Positive or Negative."
)

# 2. Define the path (Must match the save location in train.py)
MODEL_PATH = "models/sentiment_pipeline.joblib"

# 3. Load the model
try:
    # Use the pipeline saved by train.py
    sentiment_pipeline = joblib.load(MODEL_PATH)
    print("INFO: Model loaded successfully.")
    model_loaded_successfully = True
except FileNotFoundError:
    # This will print to the Uvicorn log if the model wasn't saved correctly by train.py
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}.")
    sentiment_pipeline = None
    model_loaded_successfully = False

# 4. Define a basic root endpoint
@app.get("/")
def read_root():
    """Returns status of the API and model loading."""
    if model_loaded_successfully:
        return {"message": "Sentiment Analysis API is running!", "model_status": "Loaded"}
    else:
        return {"message": "API is running but model failed to load!", "model_status": "Error"}

# 5. Define the prediction endpoint
# We use ReviewInput as the request body to enforce structure
@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    """Predicts sentiment (Positive/Negative) for a given review text."""
    
    review = data.review # Extract the review string from the Pydantic model
    
    if sentiment_pipeline is None:
        return {"error": "Model failed to load. Please run train.py first.", "review": review}
        
    # The pipeline expects a list or Series of strings
    prediction = sentiment_pipeline.predict(pd.Series([review]))[0]
    
    # Calculate probability/confidence for the predicted class
    # predict_proba returns [prob_negative, prob_positive]
    probabilities = sentiment_pipeline.predict_proba(pd.Series([review]))[0]
    confidence = probabilities[prediction] # Confidence is the probability of the predicted class
    
    # Assuming labels 1=Positive, 0=Negative
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return {
        "review": review, 
        "sentiment": sentiment,
        "confidence": float(confidence) # Return confidence as float
    }