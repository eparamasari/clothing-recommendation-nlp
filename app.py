import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
import torch

# Paths to saved models and tokenizer
MODEL_PATH_BERT = "./bert_sentiment_model"
MODEL_PATH_RF = "./random_forest.pkl"
VECTORIZER_PATH = "./tfidf.pkl"

# Load the TF-IDF + Random Forest model and vectorizer
@st.cache_resource
def load_rf_model():
    rf_model = joblib.load(MODEL_PATH_RF)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return rf_model, vectorizer

# Load the BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_BERT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_BERT)
    return model, tokenizer

# Streamlit App Interface
st.title("Clothing Review Analysis")
st.write("Enter a review below to predict whether it is **Recommended** or **Not Recommended** using both the TF-IDF + Random Forest and BERT models.")

# Text input for user review
user_review = st.text_area("Enter your review text here:", "")

if st.button("Predict"):
    if user_review.strip():
        # TF-IDF + Random Forest Prediction
        rf_model, vectorizer = load_rf_model()
        tfidf_features = vectorizer.transform([user_review])  # Transform the text input
        rf_prediction = rf_model.predict(tfidf_features)[0]
        rf_confidence = max(rf_model.predict_proba(tfidf_features)[0])
        rf_sentiment = "Recommended" if rf_prediction == 1 else "Not Recommended"

        # BERT Prediction
        bert_model, bert_tokenizer = load_bert_model()
        inputs = bert_tokenizer(user_review, return_tensors="pt", truncation=True, padding=True)
        outputs = bert_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        bert_prediction = torch.argmax(probabilities, dim=1).item()
        bert_score = probabilities[0, bert_prediction].item()
        # Map the model's labels to meaningful output
        if bert_prediction in [1, 2]:
            bert_sentiment = "Not Recommended"
        elif bert_prediction in [5, 4]:
            bert_sentiment = "Recommended"
        else:
            bert_sentiment = "Neutral or Uncertain"

        # Display results
        st.subheader("Prediction Results")
        st.write("### TF-IDF + Random Forest:")
        st.write(f"**Sentiment**: {rf_sentiment}")
        st.write(f"**Confidence Score**: {rf_confidence:.2f}")

        st.write("### BERT:")
        st.write(f"**Sentiment**: {bert_sentiment}")
        st.write(f"**Confidence Score**: {bert_score:.2f}")
        st.write(f"**Sentiment intensity prediction**: {bert_prediction}")
    else:
        st.error("Please enter some review text for prediction.")
