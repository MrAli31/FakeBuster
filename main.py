import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import sklearn
import warnings
warnings.filterwarnings('ignore')

# Load your trained model and vectorizer
try:
    model = joblib.load('my_fake_news_model.pkl')
    tfidf = joblib.load('my_tfidf_vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading model or vectorizer: {str(e)}")
    st.stop()

st.title("FakeNews Buster")
url_input = st.text_input("News URL", "http://example.com")

def predict_fake_news_by_url(url):
    # Create empty text vector (since we're not using text)
    text_vec = np.zeros((1, 500))  # Shape: (1, 500)
    
    # Ensure image features are 2D
    img_features = np.array([[2500, 100]])  # Shape: (1, 2)

    # Extract URL features
    domain = urlparse(str(url)).netloc
    is_https = 1 if 'https' in str(url) else 0
    url_features = np.array([[len(domain), is_https]])  # Shape: (1, 2)

    # Combine features
    features = np.hstack((text_vec, img_features, url_features))
    prediction = model.predict(features)[0]
    return "Fake News" if prediction == 0 else "Real News"

if st.button("Detect Fake News"):
    if url_input and url_input != "http://example.com":
        result = predict_fake_news_by_url(url_input)
        if result == "Fake News":
            st.error(f"Prediction: {result} 🚫")
        else:
            st.success(f"Prediction: {result} ✅")
    else:
        st.warning("Please provide a valid URL!")