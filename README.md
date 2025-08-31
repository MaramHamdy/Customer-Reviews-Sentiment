# Customer Reviews Sentiment Analysis

This project analyzes customer reviews and classifies their sentiment (positive / negative) using multiple approaches:
- **VADER** (baseline sentiment analysis).
- **TF-IDF + Logistic Regression**.
- **DistilBERT fine-tuned on Yelp reviews**.
- A **Streamlit app** for interactive predictions.

---

## 📂 Project Structure

```
.
├── app
│   └── streamlit_app.py         # Streamlit web app for sentiment prediction
│
├── models
│   ├── distilbert-yelp          # Fine-tuned DistilBERT checkpoints
│   │   ├── best
│   │   └── checkpoint-313
│   ├── tfidf_lr_yelp.joblib     # Saved TF-IDF + Logistic Regression model
│   └── tfidf_lr_confusion_m...  # Confusion matrix plot
│
├── notebooks
│   └── eda.py                   # Exploratory Data Analysis
│
├── screenshots                  # Screenshots of results / Streamlit app
│
├── src
│   ├── baseline_vader.py        # VADER baseline sentiment model
│   ├── data.py                  # Data preprocessing and dataset handling
│   ├── evaluate.py              # Evaluation scripts
│   ├── model_tfidf_lr.py        # TF-IDF + Logistic Regression model
│   └── model_transformer.py     # Transformer (DistilBERT) training pipeline
│
├── .gitignore
├── README.md
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MaramHamdy/customer-reviews-sentiment.git
   cd customer-reviews-sentiment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate     # On Windows
   source .venv/bin/activate  # On Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Train / Evaluate Models
Run scripts from the `src/` folder:
```bash
python src/data.py
python src/baseline_vader.py
python src/model_tfidf_lr.py
python src/model_transformer.py
python src/evaluate.py
```

### Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

This will launch a web app in your browser where you can input customer reviews and see sentiment predictions.

---

## 📊 Models

- **Baseline (VADER)** – Rule-based sentiment analyzer.
![VADER Accuracy](Screenshots/VADER.png)
- **TF-IDF + Logistic Regression** – Classical ML model.
![TF-IDF + Logistic Regression Accuracy](Screenshots/TF_IDF+LR.png)
- **DistilBERT** – Transformer model fine-tuned on Yelp dataset.
![DistilBERT Accuracy](screenshots\model_transformer.png)

---

## ✅ Evaluate

- **Best Model** - **DistilBERT** with the best accuracy that = 0.941

---

## 📷 Streamlit Web Screenshots
![Streamlit Screenshot](Screenshots/app_positive.png)

---

## 📝 License
MIT License
