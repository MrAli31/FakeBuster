import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import sklearn

# Print sklearn version for debugging
st.write(f"Using scikit-learn version: {sklearn.__version__}")

# Load your trained model and vectorizer
try:
    model = joblib.load('my_fake_news_model.pkl')
    tfidf = joblib.load('my_tfidf_vectorizer.pkl')
    st.success("Model and vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {str(e)}")
    st.stop()

st.title("My Fake News Detector")
title_input = st.text_input("News Title", "Enter title...")
text_input = st.text_area("News Article Text", "Enter article text...")
url_input = st.text_input("News URL", "http://example.com")

def predict_fake_news(title, text, url):
    # Combine and clean text
    full_text = str(title) + ' ' + str(text)
    clean_text = re.sub(r'[^a-zA-Z\s]', '', full_text.lower())
    text_vec = tfidf.transform([clean_text]).toarray()  # Shape: (1, 500)

    # Ensure image features are 2D
    img_features = np.array([[2500, 100]])  # Shape: (1, 2)

    # Extract and ensure URL features are 2D
    domain = urlparse(str(url)).netloc
    is_https = 1 if 'https' in str(url) else 0
    url_features = np.array([[len(domain), is_https]])  # Shape: (1, 2)

    # Combine features (all 2D arrays)
    features = np.hstack((text_vec, img_features, url_features))  # Shape: (1, 504)
    prediction = model.predict(features)[0]
    return "Fake News" if prediction == 0 else "Real News"

if st.button("Detect Fake News"):
    if title_input and text_input and url_input:
        result = predict_fake_news(title_input, text_input, url_input)
        if result == "Fake News":
            st.error(f"Prediction: {result} 🚫")
        else:
            st.success(f"Prediction: {result} ✅")
    else:
        st.warning("Please provide title, text, and URL!")