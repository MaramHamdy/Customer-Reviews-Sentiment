# app/streamlit_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

@st.cache_resource
def load_pipe(model_dir="models/distilbert-yelp/best"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return TextClassificationPipeline(
        model=model, tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=True
    )

st.title("Customer Reviews Sentiment 🌟")
st.write("Type/paste a review and get a sentiment prediction.")

pipe = load_pipe()
txt = st.text_area("Review text", height=160, placeholder="e.g., The delivery was fast and the product quality is excellent!")

if st.button("Analyze"):
    if txt.strip():
        scores = pipe(txt)[0]
        # label ids assumed 0=NEGATIVE, 1=POSITIVE
        pos = next(s['score'] for s in scores if s['label'].endswith("1") or s['label'].upper().startswith("POS"))
        neg = next(s['score'] for s in scores if s['label'].endswith("0") or s['label'].upper().startswith("NEG"))
        pred = "Positive ✅" if pos >= neg else "Negative ❌"
        st.subheader(pred)
        st.write({s['label']: round(s['score'], 4) for s in scores})
    else:
        st.warning("Please enter some text.")

st.caption("Model: DistilBERT fine-tuned on Yelp Polarity (subset).")
