# src/ui.py - FIXED CODE
import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Sentiment Analyzer")
st.write("Enter text and get a Positive/Negative prediction")

text = st.text_area("Enter your review here:")

if st.button("Predict"):
    if text.strip():
        # Call backend API
        url = "http://127.0.0.1:8000/predict"
        
        # --- FIX 1: Change JSON payload key from 'text' to 'review' ---
        # The FastAPI endpoint expects a variable named 'review'
        res = requests.post(url, json={"review": text})
        
        if res.status_code == 200:
            output = res.json()
            
            # --- FIX 2: Change expected key from 'label' to 'sentiment' ---
            st.success(f"Prediction: {output['sentiment']}")
            
            # --- FIX 3: Removed confidence line as FastAPI doesn't return it ---
            # You can add the confidence back if you update the FastAPI app.
            
        else:
            # Display detailed error if connection is made but prediction fails
            st.error(f"Error calling API: Status Code {res.status_code}")
            try:
                st.write("API Response:", res.json())
            except:
                st.write("API Response:", res.text)
    else:
        st.warning("Please enter some text.")