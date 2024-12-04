import pickle
from transformers import pipeline

# Load models
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("random_forest.pkl", "rb") as f:
    rf = pickle.load(f)

bert_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

def predict_tfidf_rf(text):
    text_tfidf = tfidf.transform([text])
    prediction = rf.predict(text_tfidf)
    return "Recommend" if prediction[0] == 1 else "Not Recommend"

def predict_bert(text):
    result = bert_model(text)
    prediction = 1 if result[0]['label'] in ["5 stars", "4 stars"] else 0
    return "Recommend" if prediction == 1 else "Not Recommend"
