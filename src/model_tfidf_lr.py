# src/model_tfidf_lr.py
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

def get_xy(split="train"):
    ds = load_dataset("yelp_polarity")
    X = [ex["text"] for ex in ds[split]]
    y = [ex["label"] for ex in ds[split]]
    return X, y

def main():
    X_train, y_train = get_xy("train")
    X_test, y_test = get_xy("test")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print("TF-IDF + LR")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, digits=4))

    # save
    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/tfidf_lr_yelp.joblib")

if __name__ == "__main__":
    main()
