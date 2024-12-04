# **Clothing Review Analysis**

This project is a web-based application that performs sentiment analysis on clothing reviews. Users can input review text, and the app predicts whether the review is **Recommended** or **Not Recommended** using two models:
1. **TF-IDF + Random Forest**
2. **BERT (Bidirectional Encoder Representations from Transformers)**

The app provides side-by-side predictions and confidence scores for comparison.

https://github.com/user-attachments/assets/a42bf14b-7c69-4eb8-bbfc-fb8c6c641142

---

### **Features**

- **Two Models for Prediction**:
  - **TF-IDF + Random Forest**: Lightweight, interpretable traditional machine learning model.
  - **BERT**: Advanced deep learning-based NLP model for high accuracy.
  
- **Streamlit Interface**:
  - Simple web UI to enter reviews for prediction.
  - Display results for both models with confidence scores.

---

### **Tech Stack**

- **Backend**:
  - Python
  - Hugging Face `transformers`
  - Scikit-learn

- **Web Framework**:
  - Streamlit

- **Machine Learning Models**:
  - Pretrained BERT model fine-tuned for sentiment classification.
  - Random Forest classifier trained on TF-IDF features.

---

### **Setup Instructions**

#### **1. Copy and extract the zip file**

#### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **3. Prepare Model Files**

- Place the **TF-IDF + Random Forest** model and vectorizer in the root directory:
  - `tfidf_rf_model.pkl`
  - `tfidf_vectorizer.pkl`

- Save the fine-tuned **BERT model** into the directory `bert_sentiment_model/`:
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer_config.json`
  - `vocab.txt`

---

### **Run the App**

To start the Streamlit app, use the following command:
```bash
streamlit run app.py
```

Once running, open the link provided in the terminal to access the web app in your browser.

---

### **Usage**

1. **Enter Review Text**: In the text area provided, input a review you want to analyze.
2. **Get Predictions**: Click on the "Predict" button.
3. **View Results**: The app displays:
   - Predicted sentiment (`Recommended` or `Not Recommended`).
   - Confidence scores from both the **TF-IDF + Random Forest** and **BERT** models.

---

### **Directory Structure**

```
sentiment-analysis-app/
├── app.py                # Main Streamlit app script
├── tfidf_rf_model.pkl    # Trained Random Forest model
├── tfidf_vectorizer.pkl  # Trained TF-IDF vectorizer
├── bert_sentiment_model/ # Directory containing the saved BERT model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

### **Model Details**

#### **TF-IDF + Random Forest**
- Traditional machine learning pipeline.
- Uses TF-IDF for feature extraction and a Random Forest classifier for predictions.

#### **BERT**
- Fine-tuned `bert-base-uncased` model from Hugging Face's `transformers`.
- Tokenizer processes the review text, and the model predicts sentiment.

---

### **Future Enhancements**

1. **Batch Predictions**: Add support for analyzing multiple reviews via file upload.
2. **Custom Threshold**: Allow users to set a confidence threshold for predictions.
3. **Visualization**: Include charts or graphs for a better understanding of model outputs.

---

### How to Train Models

Run the `train.py` script to train the **TF-IDF + Random Forest** model and prepare the **BERT pipeline**:
```bash
python train.py
```
This will save the trained models (`tfidf.pkl`, `random_forest.pkl`) for the TF-IDF + Random Forest method.

---

### **Acknowledgements**

- Hugging Face Transformers for pre-trained BERT models.
- Kaggle for the dataset: "Women's Clothing E-Commerce Reviews".

---

### **Contributors**

- **Ernitia Paramasari**  
  Data Scientist and Machine Learning Engineer

Feel free to contribute to this project by submitting issues or pull requests!

---

### **License**

This project is licensed under the [MIT License](LICENSE).  

Enjoy analyzing clothing reviews with cutting-edge NLP models! 🎉
