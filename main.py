
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="FakeNews Detector", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .status-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('my_fake_news_model.pkl')
        tfidf = joblib.load('my_tfidf_vectorizer.pkl')
        return model, tfidf
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, tfidf = load_models()

def preprocess_text(text):
    # Basic text cleaning
    text = text.lower()
    text = text.strip()
    return text

def get_prediction(text, is_url=False):
    text = preprocess_text(text)
    
    if is_url:
        parsed_url = urlparse(text)
        domain = parsed_url.netloc.lower()
        
        # Reliable domains check
        reliable_domains = ['reuters.com', 'apnews.com', 'bbc.com', 'npr.org', 'nytimes.com']
        if any(domain.endswith(rd) for rd in reliable_domains):
            return 1, 0.95
            
        # Suspicious patterns check
        suspicious = ['wp', 'wordpress', 'blog', 'free', '.tk', '.ml']
        if any(s in domain for s in suspicious):
            return 0, 0.85
    
    # Vectorize text
    text_vector = tfidf.transform([text])
    
    # Get prediction and probability
    prediction = model.predict(text_vector)[0]
    proba = model.predict_proba(text_vector)[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    
    return prediction, confidence

# UI Components
st.title("🔍 FakeNews Detector")
st.markdown("### Detect fake news using AI")

tab1, tab2 = st.tabs(["📰 Article Analysis", "🔗 URL Analysis"])

with tab1:
    article = st.text_area("Paste the article text here:", height=200)
    if st.button("Analyze Article", key="article_btn"):
        if article:
            with st.spinner("Analyzing..."):
                prediction, confidence = get_prediction(article)
                if prediction == 1:
                    st.success(f"✅ Likely Real News (Confidence: {confidence:.2%})")
                else:
                    st.error(f"⚠️ Likely Fake News (Confidence: {confidence:.2%})")
        else:
            st.warning("Please enter some text to analyze")

with tab2:
    url = st.text_input("Enter news article URL:")
    if st.button("Analyze URL", key="url_btn"):
        if url:
            with st.spinner("Analyzing..."):
                prediction, confidence = get_prediction(url, is_url=True)
                if prediction == 1:
                    st.success(f"✅ Likely Real News (Confidence: {confidence:.2%})")
                else:
                    st.error(f"⚠️ Likely Fake News (Confidence: {confidence:.2%})")
        else:
            st.warning("Please enter a URL to analyze")

# Tips section
with st.expander("📚 Tips for Spotting Fake News"):
    st.markdown("""
    - **Check the source:** Verify the website's reputation
    - **Check the date:** Old news may be recirculated out of context
    - **Check the author:** Look for the author's credentials
    - **Check the facts:** Cross-reference with other reliable sources
    - **Check your biases:** Be aware of your own confirmation bias
    """)

# Example section
st.markdown("---")
st.markdown("### Try these reliable news sources:")
cols = st.columns(3)
with cols[0]:
    st.markdown("- [Reuters](https://www.reuters.com)")
    st.markdown("- [AP News](https://www.apnews.com)")
with cols[1]:
    st.markdown("- [BBC News](https://www.bbc.com/news)")
    st.markdown("- [NPR](https://www.npr.org)")
with cols[2]:
    st.markdown("- [NY Times](https://www.nytimes.com)")
    st.markdown("- [The Guardian](https://www.theguardian.com)")
