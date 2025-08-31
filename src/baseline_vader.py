# src/baseline_vader.py
import nltk, re
from nltk.sentiment import SentimentIntensityAnalyzer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean_text(t):
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"[^A-Za-z0-9' ]+", " ", t)  # simple cleanup
    return t.strip()

def main():
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    ds = load_dataset("yelp_polarity")
    test = ds["test"]

    preds, labels = [], []
    for ex in test:
        text = clean_text(ex["text"])
        score = sia.polarity_scores(text)["compound"]
        pred = 1 if score >= 0 else 0
        preds.append(pred)
        labels.append(ex["label"])

    print("Baseline VADER")
    print("Accuracy:", accuracy_score(labels, preds))
    print("F1:", f1_score(labels, preds))
    print(classification_report(labels, preds, digits=4))

if __name__ == "__main__":
    main()
