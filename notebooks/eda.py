# notebooks/eda.py (or a short script)
from collections import Counter
from datasets import load_dataset
ds = load_dataset("yelp_polarity")
print(ds)
print("Sample:", ds["train"][0]["text"][:300])
print("Label distribution (train):", Counter([ex["label"] for ex in ds["train"]]))
