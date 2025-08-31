# src/data.py
from datasets import load_dataset

def get_dataset(name="yelp"):
    if name == "yelp":
        ds = load_dataset("yelp_polarity")
    elif name == "amazon":
        ds = load_dataset("amazon_polarity")
    else:
        raise ValueError("name must be 'yelp' or 'amazon'")
    # standardize columns
    return ds.rename_columns({"label": "label", "text": "text"})
