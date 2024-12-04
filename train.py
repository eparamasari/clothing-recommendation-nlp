import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
from transformers import pipeline

# Load data
data = pd.read_csv("data/womens-clothing-e-commerce-reviews.csv")
data = data.dropna(subset=["Review Text", "Recommended IND"])

X = data["Review Text"]
y = data["Recommended IND"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train TF-IDF + Random Forest model
print("Training TF-IDF + RandomForest...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_tfidf, y_train)

# Evaluate TF-IDF + Random Forest model
y_pred_rf = rf.predict(X_test_tfidf)
f1_rf = f1_score(y_test, y_pred_rf)
print(f"TF-IDF + RandomForest F1 Score: {f1_rf}")

# Save TF-IDF and RF model
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)

# Download BERT model
print("Downloading BERT model...")
bert_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
# Note: BERT does not require explicit training for classification.

# Evaluate BERT (Custom Evaluation)

# Map BERT sentiment labels to binary recommendation labels
def bert_to_binary(label):
    # This model typically outputs LABEL_1 (negative) to LABEL_5 (positive).
    if label in ["1 stars", "2 stars", "3 stars"]:  # Negative & neutral sentiment
        return 0
    elif label in ["4 stars", "5 stars"]:  # Positive sentiment
        return 1
    else:
        return None 

# Predict and evaluate
y_pred_bert = []
y_true_filtered = []

for text, label in zip(X_test, y_test):
    result = bert_model(text)[0]
    binary_prediction = bert_to_binary(result['label'])
    
    if binary_prediction is not None:  # Only include confident predictions
        y_pred_bert.append(binary_prediction)
        y_true_filtered.append(label)

# Calculate F1 score
f1_bert = f1_score(y_true_filtered, y_pred_bert)
print(f"BERT F1 Score: {f1_bert}")

# Save model
bert_model.save_pretrained("bert_sentiment_model")
print("Models trained and saved!")
