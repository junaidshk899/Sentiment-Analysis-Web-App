# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# --- Model Training and Saving Logic ---

# Ensure everything below this line is executed ONLY when the script is run directly
if __name__ == "__main__":
    
    # 1. Ensure 'models' directory exists inside the project root
    os.makedirs("models", exist_ok=True)
    
    # 2. Load dataset (path is relative to the project root when run with `python src/train.py`)
    df = pd.read_csv("data/reviews.csv")

    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    # 4. Define the pipeline
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
        LogisticRegression(max_iter=1000)
    )

    # 5. Train the model
    pipeline.fit(X_train, y_train)

    # 6. Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("---------------------------\n")

    # 7. Save the model to the 'models' directory inside the project root
    joblib.dump(pipeline, "models/sentiment_pipeline.joblib")
    print("✅ Model saved to models/sentiment_pipeline.joblib")