
import os
import nltk
import streamlit as st

# Install missing dependencies inside the app environment
if "nltk_resources_installed" not in st.session_state:
    os.system('pip install joblib')
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Ensure sub-resource
    st.session_state["nltk_resources_installed"] = True

import joblib
import re
from nltk.tokenize import word_tokenize

# Load model and vectorizer
@st.cache_resource
def load_pipeline():
    return joblib.load("/content/sentiment_analysis.pkl")

pipeline = load_pipeline()

# Preprocess Vietnamese text
def preprocess_vietnamese(text):
    emoji_pattern = re.compile(
        "[" +
        u"\U0001F600-\U0001F64F" +
        u"\U0001F300-\U0001F5FF" +
        u"\U0001F680-\U0001F6FF" +
        u"\U0001F1E0-\U0001F1FF" +
        u"\U00002702-\U000027B0" +
        u"\U000024C2-\U0001F251" +
        "]"
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(word_tokenize(text))
    return text

# Streamlit app
st.title("Vietnamese Sentiment Analysis")
st.write("Enter a Vietnamese review to predict its sentiment.")

review = st.text_area("Review:")

# Process input and predict
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.write("Please enter a valid review.")
    else:
        prediction = pipeline.predict([review])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Predicted Sentiment: **{sentiment}**")
