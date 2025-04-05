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
article_content = st.text_area("Or paste article content here", height=200)

def analyze_text_content(text):
    # Process and vectorize the text
    text_features = tfidf.transform([text]).toarray()
    # Add the additional features (domain length and protocol score) as zeros
    additional_features = np.zeros((text_features.shape[0], 4))
    combined_features = np.hstack((text_features, additional_features))
    prediction = model.predict(combined_features)[0]
    probability = model.predict_proba(combined_features)[0]
    confidence = probability[1] if prediction == 1 else probability[0]
    return ("Real News" if prediction == 1 else "Fake News", confidence)

def predict_fake_news_by_url(url):
    # List of known reliable news domains
    reliable_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'npr.org', 'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'wsj.com', 'bloomberg.com']
    
    # Create empty text vector (since we're not using text)
    text_vec = np.zeros((1, 500))  # Shape: (1, 500)
    
    # Parse URL and extract components
    parsed_url = urlparse(str(url))
    domain = parsed_url.netloc.lower()
    
    # Analyze web protocols and security
    is_https = 1 if parsed_url.scheme == 'https' else 0
    has_path = 1 if parsed_url.path and parsed_url.path != '/' else 0
    has_query = 1 if parsed_url.query else 0
    domain_parts = domain.split('.')
    is_subdomain = len(domain_parts) > 2
    domain_length = len(domain)
    
    # Calculate protocol score
    protocol_score = (
        is_https * 3 +  # HTTPS is important
        has_path * 1 +  # Having a specific path is good
        (0 if has_query > 3 else 1) +  # Too many query parameters might be suspicious
        (1 if is_subdomain else 2)  # Subdomains might be slightly less trustworthy
    )
    
    # Additional URL analysis
    is_known_reliable = any(rd in domain for rd in reliable_domains)
    has_news_keywords = any(kw in domain for kw in ['news', 'media', 'press'])
    
    # Calculate final reliability score
    reliability_score = (
        protocol_score +
        (5 if any(rd in domain for rd in reliable_domains) else 0) +  # Known domains
        (2 if has_news_keywords else 0) +  # News keywords
        (1 if domain_length < 30 else 0)  # Reasonable domain length
    )
    
    # Normalize the score
    normalized_score = min(reliability_score / 10.0, 1.0)
    
    # Use protocol analysis for classification
    if normalized_score > 0.7:
        return ("Real News", normalized_score)
    
    url_features = np.array([[domain_length, protocol_score]])
    
    # Combine features for unknown domains
    features = np.hstack((text_vec, np.array([[2500, 100]]), url_features))
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    confidence = probability[1] if prediction == 1 else probability[0]
    return ("Real News" if prediction == 1 else "Fake News", confidence)

if st.button("Detect Fake News"):
    if article_content:
        result, confidence = analyze_text_content(article_content)
        confidence_percentage = confidence * 100
        if result == "Real News":
            st.success(f"Prediction: {result} ✅ (Confidence: {confidence_percentage:.2f}%)")
            st.info("Tips for real news:\n- Check multiple sources\n- Look for recent updates\n- Verify the author")
        else:
            st.error(f"Prediction: {result} 🚫 (Confidence: {confidence_percentage:.2f}%)")
            st.warning("Warning signs of fake news:\n- Suspicious domain\n- Lack of HTTPS\n- Unusual URL structure")
    elif url_input and url_input != "http://example.com":
        result, confidence = predict_fake_news_by_url(url_input)
        confidence_percentage = confidence * 100
        if result == "Real News":
            st.success(f"Prediction: {result} ✅ (Confidence: {confidence_percentage:.2f}%)")
            st.info("Tips for real news:\n- Check multiple sources\n- Look for recent updates\n- Verify the author")
        else:
            st.error(f"Prediction: {result} 🚫 (Confidence: {confidence_percentage:.2f}%)")
            st.warning("Warning signs of fake news:\n- Suspicious domain\n- Lack of HTTPS\n- Unusual URL structure")
    else:
        st.warning("Please provide either a URL or article content!")
        
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