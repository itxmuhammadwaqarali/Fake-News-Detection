import streamlit as st
import joblib
import re
import string

# Load the saved models and vectorizer
LR = joblib.load('logistic_regression_model.pkl')
RFC = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Prediction function
def manual_testing(news):
    news = wordopt(news)
    new_xv_test = vectorizer.transform([news])
    
    pred_LR = LR.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    
    return f"Logistic Regression Prediction: {output_label(pred_LR[0])}, Random Forest Prediction: {output_label(pred_RFC[0])}"

# Helper function to output the label
def output_label(n):
    return "Fake News" if n == 0 else "Not Fake News"

# Set up Streamlit app
st.title("Fake News Detection")

# Handling text input
user_text = st.text_area("Enter the text for analysis:")

if user_text:
    result = manual_testing(user_text)
    st.write(result)
