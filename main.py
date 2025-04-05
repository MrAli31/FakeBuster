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
    # List of known reliable news domains
    reliable_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'npr.org', 'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'wsj.com', 'bloomberg.com']
    
    # Create empty text vector (since we're not using text)
    text_vec = np.zeros((1, 500))  # Shape: (1, 500)
    
    # Extract URL features
    domain = urlparse(str(url)).netloc.lower()
    is_https = 1 if 'https' in str(url) else 0
    domain_length = len(domain)
    
    # Additional URL analysis
    is_known_reliable = any(rd in domain for rd in reliable_domains)
    has_news_keywords = any(kw in domain for kw in ['news', 'media', 'press'])
    
    # Simple domain-based detection for reliable sources
    for reliable_domain in reliable_domains:
        if reliable_domain in domain:
            return ("Real News", 0.95)  # High confidence for trusted sources
    
    # Fall back to the scoring system for unknown domains
    reliability_score = (
        is_https * 1 +  # HTTPS: 1 point
        (2 if has_news_keywords else 0)  # News keywords: 2 points
    )
    url_features = np.array([[domain_length, reliability_score]])
    
    # Combine features for unknown domains
    features = np.hstack((text_vec, np.array([[2500, 100]]), url_features))
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    confidence = probability[1] if prediction == 1 else probability[0]
    return ("Real News" if prediction == 1 else "Fake News", confidence)

if st.button("Detect Fake News"):
    if url_input and url_input != "http://example.com":
        result, confidence = predict_fake_news_by_url(url_input)
        confidence_percentage = confidence * 100
        
        if result == "Real News":
            st.success(f"Prediction: {result} ✅ (Confidence: {confidence_percentage:.2f}%)")
            st.info("Tips for real news:\n- Check multiple sources\n- Look for recent updates\n- Verify the author")
        else:
            st.error(f"Prediction: {result} 🚫 (Confidence: {confidence_percentage:.2f}%)")
            st.warning("Warning signs of fake news:\n- Suspicious domain\n- Lack of HTTPS\n- Unusual URL structure")
    else:
        st.warning("Please provide a valid URL!")
        
st.markdown("---")
st.markdown("### Try these example URLs:")
st.markdown("#### Reliable News Sources:")
st.markdown("- Reuters: https://www.reuters.com")
st.markdown("- AP News: https://www.apnews.com")
st.markdown("- BBC News: https://www.bbc.com/news")
st.markdown("- NPR: https://www.npr.org")
st.markdown("- The Guardian: https://www.theguardian.com")
st.markdown("- Bloomberg: https://www.bloomberg.com")

st.markdown("#### Example Article URLs:")
st.markdown("- Reuters Article: https://www.reuters.com/world/")
st.markdown("- AP News Article: https://apnews.com/hub/us-news")
st.markdown("- BBC Article: https://www.bbc.com/news/world")