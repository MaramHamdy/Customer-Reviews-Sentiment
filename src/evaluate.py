# src/evaluate.py
import joblib
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt

def main():
    ds = load_dataset("yelp_polarity")
    X_test = [ex["text"] for ex in ds["test"]]
    y_test = [ex["label"] for ex in ds["test"]]

    pipe = joblib.load("models/tfidf_lr_yelp.joblib")
    preds = pipe.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, digits=4))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("TF-IDF + LR on Yelp Polarity")
    plt.tight_layout()
    plt.savefig("models/tfidf_lr_confusion_matrix.png")
    print("Saved confusion matrix to models/tfidf_lr_confusion_matrix.png")

if __name__ == "__main__":
    main()
