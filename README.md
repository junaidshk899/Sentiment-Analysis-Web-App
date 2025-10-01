# 📝 Sentiment Analysis Web App

A beginner-friendly **Machine Learning project** that performs **Sentiment Analysis** on text (positive/negative) using **Logistic Regression** with TF-IDF features.  

The project includes:  
- A training pipeline (`train.py`) to build and save the model.  
- A FastAPI backend (`app.py`) that serves predictions as an API.  
- A Streamlit frontend (`ui.py`) that allows users to test the model in a user-friendly interface.  
- Deployed locally, but ready for cloud deployment (Heroku, Render, etc.).  

---

## 🚀 Features
- Train your own sentiment model with a simple dataset (`data/reviews.csv`)  
- Save and reuse the trained pipeline (`models/sentiment_pipeline.joblib`)  
- REST API with FastAPI for predictions (`/predict`)  
- Interactive web UI with Streamlit  
- Beginner-friendly and extendable  

## 2️⃣ Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

## 3️⃣ Install dependencies
pip install -r requirements.txt

### Usage
## 🔹 1. Train the model

Make sure data/reviews.csv exists with text and label columns.

python src/train.py


This saves the model into:

models/sentiment_pipeline.joblib

## 🔹 2. Run the FastAPI backend

Start the API server:

uvicorn src.app:app --reload --port 8000

## 🔹 3. Run the Streamlit UI
streamlit run src/ui.py


This launches a simple, interactive web app in your browser where you can enter text and see sentiment predictions.

## 📊 Example Predictions

Input: "This movie was absolutely amazing!"
Output: Positive ✅

Input: "The acting was horrible and boring."
Output: Negative ❌

## 🔮 Future Improvements

Use larger datasets (IMDB, Yelp, Amazon Reviews) for better accuracy

Deploy on cloud platforms (Heroku, Render, Hugging Face Spaces)

Add a Neutral category for 3-class sentiment analysis

Visualize training metrics and predictions in Streamlit
